// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest-param-test.h>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {
using InputShapes = std::vector<InputShape>;
using PagedAttnTestParams = std::tuple<ElementType, InputShapes>;

class SdpaSinkTest : public testing::WithParamInterface<PagedAttnTestParams>,
                           virtual public ov::test::SubgraphBaseTest,
                           public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnTestParams>& obj) {
        const auto& [inType, inputShapes] = obj.param;
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "Prc=" << inType;

        return result.str();
    }
    static std::shared_ptr<ov::op::v0::Parameter> make_param(const PartialShape& pshape,
                                                             element::Type element_type,
                                                             const std::string& name) {
        auto param = std::make_shared<v0::Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }
    virtual std::shared_ptr<ov::Model> get_model(ov::element::Type data_type,
                                                     ov::Dimension::value_type head_size = 64,
                                                     ov::Dimension::value_type head_num = 8) {
        // q, k, v use L,B,H,S layout
        ov::PartialShape q_shape, kv_shape, past_shape, atten_mask_shape, scale_shape, sink_shape;
        ov::ParameterVector inputParams;
        past_shape = {-1, 1, head_num, head_size};
        q_shape = {-1, 1, static_cast<int64_t>(head_num), head_size};
        kv_shape = {-1, 1, head_num, head_size};
        atten_mask_shape = {1, head_num, -1, -1};
        scale_shape = {1};
        sink_shape = {1, head_num, 1, 1};

        auto q = make_param(q_shape, data_type, "q");
        auto k = make_param(kv_shape, data_type, "k");
        auto v = make_param(kv_shape, data_type, "v");
        auto past_kv = make_param(past_shape, data_type, "past_kv");
        auto atten_mask = make_param(atten_mask_shape, data_type, "atten_mask");
        auto scale = make_param(scale_shape, data_type, "scale");
        auto sink = make_param(sink_shape, data_type, "sink");
        inputParams.push_back(q);
        inputParams.push_back(k);
        inputParams.push_back(v);
        inputParams.push_back(atten_mask);
        inputParams.push_back(scale);
        inputParams.push_back(sink);
        inputParams.push_back(past_kv);
        auto var_k =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, data_type, "pastk"});
        auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[6], var_k);
        pastk->set_friendly_name("pastk_r");
        auto var_v =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, data_type, "pastv"});
        auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[6], var_v);
        pastv->set_friendly_name("pastv_r");
        std::vector<size_t> transposeOrder{1, 2, 0, 3};
        auto preOrder = op::v0::Constant::create(ov::element::i32, {4}, transposeOrder);
        std::shared_ptr<ov::Node> q_in = std::make_shared<ov::op::v1::Transpose>(inputParams[0], preOrder);

        auto concat_axis = transposeOrder[2];
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        beam_idx->set_friendly_name("beam_idx");
        inputParams.push_back(beam_idx);
        auto gatherK =
            std::make_shared<ov::op::v8::Gather>(pastk,
                                                 beam_idx,
                                                 op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
        auto gatherV =
            std::make_shared<ov::op::v8::Gather>(pastv,
                                                 beam_idx,
                                                 op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
        auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, concat_axis);
        auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, concat_axis);
        std::shared_ptr<ov::Node> k_in = concatK;
        std::shared_ptr<ov::Node> v_in = concatV;
        k_in = std::make_shared<ov::op::v1::Transpose>(k_in, preOrder);
        v_in = std::make_shared<ov::op::v1::Transpose>(v_in, preOrder);
        auto sdp = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_in, k_in, v_in, atten_mask, scale, sink, false);
        sdp->set_friendly_name("mha");
        auto pastk_assign = std::make_shared<ov::op::v6::Assign>(concatK, var_k);
        auto pastv_assign = std::make_shared<ov::op::v6::Assign>(concatV, var_v);
        pastk_assign->set_friendly_name("pastk_w");
        pastv_assign->set_friendly_name("pastv_w");
        auto get_reshape_order = [](const ov::PartialShape& qkv_shape,
                                    const std::vector<size_t>& transposeOrder) -> std::vector<size_t> {
            assert(transposeOrder.size() == 4);
            auto H = qkv_shape[transposeOrder[1]].get_length();
            auto S = qkv_shape[transposeOrder[3]].get_length();
            return std::vector<size_t>{0, static_cast<size_t>(H * S)};
        };
        const auto reshapeOrder = get_reshape_order(q_shape, transposeOrder);
        auto postOrder =
            ov::op::v0::Constant::create(ov::element::i32, {4}, std::vector<size_t>{2, 0, 1, 3});  // BHLS -> LBHS
        auto transposeSDP = std::make_shared<ov::op::v1::Transpose>(sdp, postOrder);

        auto constReshape = ov::op::v0::Constant::create(ov::element::i32, {2}, reshapeOrder);
        auto reshapeSDP =
            std::make_shared<ov::op::v1::Reshape>(transposeSDP,
                                                  constReshape,
                                                  true);  // use LBHS to better compare data between pa and sdpa
        SinkVector sinks{pastk_assign, pastv_assign};
        ov::OutputVector results{reshapeSDP};
        auto model = std::make_shared<Model>(results, sinks, inputParams, "sdpa_model");
        return model;
    }

    void SetUp() override {
        const auto& [inType, inputShapes] = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        if (inType == ElementType::bf16) {
            configuration[ov::hint::inference_precision.name()] = ov::element::bf16;
            configuration[ov::hint::kv_cache_precision.name()] = ov::element::f16;
            rel_threshold = 0.02f;
            abs_threshold = 0.02f;
        } else if (inType == ElementType::f32) {
            configuration[ov::hint::kv_cache_precision.name()] = ov::element::f32;
        } else if (inType == ElementType::f16) {
            configuration[ov::hint::kv_cache_precision.name()] = ov::element::f16;
        }
        init_input_shapes(inputShapes);
        ov::ParameterVector inputParams;

        function = get_model(inType, 64, 8);
    }

    virtual void generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        auto create_input = [this](std::shared_ptr<ov::op::v0::Parameter> param, ov::Shape shape, float val = 0) {
            if (param->get_element_type() == ov::element::i32) {
                ov::Tensor t{ov::element::i32, shape};
                auto size = shape[0];
                auto* p = static_cast<int*>(t.data());
                auto start = static_cast<int>(val);
                for (size_t i = 0; i < size; i++) {
                    p[i] = (start + i) % size;
                }
                inputs.insert({param, t});
            } else if (param->get_element_type() == ov::element::f32) {
                ov::Tensor t{ov::element::f32, shape};
                utils::fill_data_random(static_cast<float*>(t.data()), t.get_size(), 2, -1, 10);
                inputs.insert({param, t});
            } else if (param->get_element_type() == ov::element::f16) {
                ov::Tensor t{ov::element::f16, shape};
                utils::fill_data_random(static_cast<ov::float16*>(t.data()), t.get_size(), 2, -1, 10);
                inputs.insert({param, t});
            } else {
                ASSERT_TRUE(param->get_element_type() == ov::element::bf16);
                ov::Tensor t{ov::element::bf16, shape};
                utils::fill_data_random(static_cast<ov::bfloat16*>(t.data()), t.get_size(), 2, -1, 10);
                inputs.insert({param, t});
            }
        };

        // q, k, v, pastkv
        create_input(function->get_parameters()[0], targetInputStaticShapes[0]);
        create_input(function->get_parameters()[1], targetInputStaticShapes[0]);
        create_input(function->get_parameters()[2], targetInputStaticShapes[0]);
        create_input(function->get_parameters()[3], targetInputStaticShapes[2]);
        create_input(function->get_parameters()[4],
                     function->get_parameters()[4]->get_partial_shape().to_shape());
        create_input(function->get_parameters()[5],
                     function->get_parameters()[5]->get_partial_shape().to_shape());
        create_input(function->get_parameters()[6], targetInputStaticShapes[1]);
        create_input(function->get_parameters()[7], ov::Shape{targetInputStaticShapes[0][1]});
    }
    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }
    void reset() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    }
    void run_test(std::shared_ptr<ov::Model> model) {
        function = model;
        prepare();

        auto core = ov::test::utils::PluginCache::get().core();
        ov::AnyMap configRef = {{"DISABLE_TRANSFORMATIONS" , "YES"}};
        auto compiledModelRef = core->compile_model(model,
                ov::test::utils::DEVICE_TEMPLATE, configRef);
        auto inferRequestRef = compiledModelRef.create_infer_request();

        int idx = 0;
        for (auto&& shapes : targetStaticShapes) {
            generate(idx++, shapes);
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
                inferRequestRef.set_tensor(input.first, input.second);
            }
            inferRequest.infer();
            inferRequestRef.infer();
            auto logits = inferRequest.get_output_tensor(0);
            auto logitsRef = inferRequestRef.get_output_tensor(0);
            ov::test::utils::compare(logitsRef, logits, abs_threshold, rel_threshold);
        }
        reset();
        for (auto&& state : inferRequestRef.query_state()) {
            state.reset();
        }
    }
};

TEST_P(SdpaSinkTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, inputShapes] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    run_test(function);
}

namespace {

const std::vector<InputShapes> inputShapes = {  // greedy search
    {
        // q k v
        {{-1, 1, 8, 64}, {{10, 1, 8, 64}, {1, 1, 8, 64}}},
        // pask kv
        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {10, 1, 8, 64}}},
        // attention_mask
        {{1, 8, -1, -1}, {{1, 8, 10, 10}, {1, 8, 1, 11}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_SdpaSinkTest,
                         SdpaSinkTest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::f16, ElementType::bf16),
                                            ::testing::ValuesIn(inputShapes)),
                         SdpaSinkTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
