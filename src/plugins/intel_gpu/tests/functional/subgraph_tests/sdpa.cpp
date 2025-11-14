// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/opsets/opset13_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/common_optimizations/sdpa_fusion.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace {
// validate the batch axis padding for sdpa_micro kernel.
class SDPA : virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        {
            auto capabilities = core->get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
            if (std::find(capabilities.cbegin(), capabilities.cend(), ov::intel_gpu::capability::HW_MATMUL) == capabilities.cend())
                GTEST_SKIP();
        }
        auto inType = ov::element::f16;
        ov::Shape inputShape{3, 4, 8, 16};
        auto constant1 = ov::op::v0::Constant::create(ov::element::i32, {4}, {1, 4, 8, 16});
        auto constant2 = ov::op::v0::Constant::create(ov::element::i32, {4}, {1, 4, 8, 16});
        auto constant3 = ov::op::v0::Constant::create(ov::element::i32, {4}, {1, 4, 8, 16});
        auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputShape);
        auto split_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32, ov::Shape{}, std::vector<int64_t>{0});
        auto split = std::make_shared<ov::op::v1::Split>(input, split_axis_op, 3);

        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(split->output(0), constant1, false);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(split->output(1), constant2, false);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(split->output(2), constant3, false);
        auto sdpa = std::make_shared<ov::opset13::ScaledDotProductAttention>(reshape1, reshape2, reshape3, false);
        sdpa->set_friendly_name("sdpa");

        auto output = std::make_shared<ov::op::v0::Result>(sdpa->output(0));
        function = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input}, "sdpa_model");

        functionRefs = function->clone();
        ov::pass::Manager manager;

        // Decompose ScaledDotProductAttention
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);

        bool has_long_seq = inputShape[2] >= 384 || inputShape[3] >= 128;
        if (inType == ov::element::f16) {
            if (has_long_seq) {
                abs_threshold = 0.025;
                rel_threshold = 0.025;
            } else {
                abs_threshold = 0.005;
                rel_threshold = 0.005;
            }
        }
    }
};

class SDPAFusion : virtual public ov::test::SubgraphBaseStaticTest,
                   public testing::WithParamInterface<std::tuple<ov::PartialShape,  // query shape
                                                                 ov::Shape,         // query reshape shape
                                                                 ov::PartialShape,  // key shape
                                                                 ov::Shape,         // key reshape shape
                                                                 ov::PartialShape,  // value shape
                                                                 ov::Shape,         // value reshape shape
                                                                 ov::PartialShape,  // mask shape
                                                                 float,             // scale value
                                                                 float,             // abs_threshold
                                                                 float>             // rel_threshold
                                                      > {
protected:
    void create_model() {
        auto params = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;
        inType = ov::element::f16;
        bool reshape = false;

        const ov::PartialShape query_shape = std::get<0>(params);
        const ov::Shape query_reshape_shape = std::get<1>(params);
        const ov::PartialShape key_shape = std::get<2>(params);
        const ov::Shape key_reshape_shape = std::get<3>(params);
        const ov::PartialShape value_shape = std::get<4>(params);
        const ov::Shape value_reshape_shape = std::get<5>(params);
        const ov::PartialShape attention_mask_shape = std::get<6>(params);
        const ov::Shape scale_shape{1};

        const auto query = std::make_shared<ov::op::v0::Parameter>(inType, query_shape);
        std::shared_ptr<ov::op::v1::Reshape> query_reshaped;
        if (query_shape != query_reshape_shape) {
            const auto query_reshape_params = ov::op::v0::Constant::create(ov::element::i64,
                                                                           ov::Shape{query_reshape_shape.size()},
                                                                           query_reshape_shape);
            query_reshaped = std::make_shared<ov::op::v1::Reshape>(query, query_reshape_params, true);
            reshape = true;
        }

        const auto key = std::make_shared<ov::op::v0::Parameter>(inType, key_shape);
        std::shared_ptr<ov::op::v1::Reshape> key_reshaped;
        if (key_shape != key_reshape_shape) {
            const auto key_reshape_params =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{key_reshape_shape.size()}, key_reshape_shape);
            key_reshaped = std::make_shared<ov::op::v1::Reshape>(key, key_reshape_params, true);
            reshape = true;
        }

        const auto value = std::make_shared<ov::op::v0::Parameter>(inType, value_shape);
        std::shared_ptr<ov::op::v1::Reshape> value_reshaped;
        if (value_shape != value_reshape_shape) {
            const auto value_reshape_params = ov::op::v0::Constant::create(ov::element::i64,
                                                                           ov::Shape{value_reshape_shape.size()},
                                                                           value_reshape_shape);
            value_reshaped = std::make_shared<ov::op::v1::Reshape>(value, value_reshape_params, true);
            reshape = true;
        }

        const auto mask = std::make_shared<ov::op::v0::Parameter>(inType, attention_mask_shape);
        const auto scale_const = ov::op::v0::Constant::create(inType, {}, std::vector<float>{std::get<7>(params)});
        std::shared_ptr<ov::op::v0::MatMul> qk;
        if (reshape) {
            qk = std::make_shared<ov::op::v0::MatMul>(query_reshaped, key_reshaped, false, true);
        } else {
            qk = std::make_shared<ov::op::v0::MatMul>(query, key, false, true);
        }

        const auto scaled_qk = std::make_shared<ov::op::v1::Multiply>(qk, scale_const);
        const auto mask_add = std::make_shared<ov::op::v1::Add>(scaled_qk, mask);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(mask_add, -1);
        std::shared_ptr<ov::op::v0::MatMul> qkv;
        std::shared_ptr<ov::op::v1::Reshape> qkv_reshaped;
        std::shared_ptr<ov::op::v0::Result> output;
        if (reshape) {
            qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value_reshaped, false, false);
            const auto qkv_reshape_params =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{query_shape.size()}, query_shape.to_shape());
            qkv_reshaped = std::make_shared<ov::op::v1::Reshape>(qkv, qkv_reshape_params, true);
            output = std::make_shared<ov::op::v0::Result>(qkv_reshaped->output(0));
        } else {
            qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);
            output = std::make_shared<ov::op::v0::Result>(qkv->output(0));
        }

        function = std::make_shared<ov::Model>(ov::OutputVector{output},
                                               ov::ParameterVector{query, key, value, mask},
                                               "sdpa_model");

        functionRefs = function->clone();

        abs_threshold = std::get<8>(params);
        rel_threshold = std::get<9>(params);
    }

    void check_results() {
        auto exec_model = compiledModel.get_runtime_model();

        int fused_node_found = 0;
        for (const auto& n : exec_model->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == "scaled_dot_product_attention")
                fused_node_found++;
        }
        ASSERT_EQ(fused_node_found, 1);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        auto itTargetShape = targetInputStaticShapes.begin();
        for (const auto& param : function->get_parameters()) {
            std::shared_ptr<ov::Node> inputNode = param;
            for (size_t i = 0; i < param->get_output_size(); i++) {
                for (const auto& node : param->get_output_target_inputs(i)) {
                    std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                    for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                        if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                            const auto& tensor = ov::test::utils::create_and_fill_tensor(
                                inType,
                                *itTargetShape,
                                ov::test::utils::InputGenerateData(0, 8, 32, 1));
                            inputs.insert({param, tensor});
                            break;
                        }
                    }
                }
            }
            itTargetShape++;
        }
    }
};

TEST_F(SDPA, smoke_Inference) {
    run();
}

TEST_P(SDPAFusion, Inference) {
    create_model();
    run();

    check_results();
}

INSTANTIATE_TEST_SUITE_P(SDPAFusionTests,
                         SDPAFusion,
                         ::testing::Values(std::make_tuple(ov::PartialShape{10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{1024, 77},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f),
                                            std::make_tuple(ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{10, 1024, 77},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f),
                                           std::make_tuple(ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{10, 1024, 1024},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f),
                                           std::make_tuple(ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{77, 77},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f),
                                           std::make_tuple(ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{1, 10, 1024, 64},
                                                           ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{1, 10, 1024, 64},
                                                           ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{1, 10, 1024, 64},
                                                           ov::PartialShape{10, 1024, 1024},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f)));
}  // namespace
