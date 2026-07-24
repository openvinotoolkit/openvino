// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
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
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/variadic_split.hpp"

#include "openvino/opsets/opset13_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/common_optimizations/sdpa_fusion.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include <numeric>

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

// Validate that non-PA SDPA with f16 K/V inputs is not incorrectly blocked
// from using micro kernel when KV_CACHE_PRECISION is globally set to u4.
// This simulates a Vision Encoder SDPA node running alongside a PA-based LLM
// that uses INT4 KV cache. The global config should not affect the Vision Encoder path.
class SDPAWithInt4KVCacheConfig : public SDPA {
protected:
    void SetUp() override {
        SDPA::SetUp();
        configuration.insert(ov::hint::kv_cache_precision(ov::element::u4));
    }
};

TEST_F(SDPAWithInt4KVCacheConfig, smoke_Inference) {
    run();
}

class SDPAFusion : virtual public ov::test::SubgraphBaseStaticTest,
                   public testing::WithParamInterface<std::tuple<ov::PartialShape,  // 0: query shape
                                                                 ov::Shape,         // 1: query reshape shape
                                                                 ov::PartialShape,  // 2: key shape
                                                                 ov::Shape,         // 3: key reshape shape
                                                                 ov::PartialShape,  // 4: value shape
                                                                 ov::Shape,         // 5: value reshape shape
                                                                 ov::PartialShape,  // 6: mask shape
                                                                 float,             // 7: scale value
                                                                 float,             // 8: abs_threshold
                                                                 float,             // 9: rel_threshold
                                                                 bool,              // 10: is_complex_gqa
                                                                 float>>            // 11: kv_num_head_factor
                                                         {
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
        bool is_complex_gqa =  std::get<10>(params);
        float kv_num_head_factor = std::get<11>(params);

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

        std::shared_ptr<ov::Node> key_input = reshape ? std::static_pointer_cast<ov::Node>(key_reshaped) : std::static_pointer_cast<ov::Node>(key);
        std::shared_ptr<ov::Node> value_input = reshape ? std::static_pointer_cast<ov::Node>(value_reshaped) : std::static_pointer_cast<ov::Node>(value);

        ov::ParameterVector model_params = {query, key, value};

        if (is_complex_gqa) {
            auto q_shape = query_shape.to_shape();                  // [1, 8, 10, 256]
            auto k_shape = key_shape.to_shape();                    // [1, 1, 10, 256]
            auto mask_shape = attention_mask_shape.to_shape();      // [10, 842]

            // Deduce past sequence length from the mask size
            size_t total_seq_len = mask_shape[1];                   // 842
            size_t current_seq_len = k_shape[2];                    // 10
            size_t past_seq_len = total_seq_len - current_seq_len;  // 832

            // Create past_key / past_value parameters dynamically
            ov::Shape past_shape = { k_shape[0], k_shape[1], past_seq_len, k_shape[3] };
            auto past_key = std::make_shared<ov::op::v0::Parameter>(inType, past_shape);
            auto past_value = std::make_shared<ov::op::v0::Parameter>(inType, past_shape);
            model_params.push_back(past_key);
            model_params.push_back(past_value);

            // 1. Concat
            auto concat_k = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{ past_key, key_input }, 2);
            auto concat_v = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{ past_value, value_input }, 2);

            // 2. Reshape to 5D -> [1, 1, 1, 842, 256]
            ov::Shape unsqueeze_shape = { q_shape[0], k_shape[1], 1, total_seq_len, k_shape[3] };
            auto reshape1_k_const = ov::op::v0::Constant::create(ov::element::i64, { unsqueeze_shape.size() }, unsqueeze_shape);
            auto reshape1_v_const = ov::op::v0::Constant::create(ov::element::i64, { unsqueeze_shape.size() }, unsqueeze_shape);
            auto reshape1_k = std::make_shared<ov::op::v1::Reshape>(concat_k, reshape1_k_const, true);
            auto reshape1_v = std::make_shared<ov::op::v1::Reshape>(concat_v, reshape1_v_const, true);

            // 3. Broadcast to match Query heads -> [1, 1, 8, 842, 256]
            ov::Shape broadcast_shape = { q_shape[0], k_shape[1], q_shape[1], total_seq_len, k_shape[3] };
            auto broadcast_k_const = ov::op::v0::Constant::create(ov::element::i64, { broadcast_shape.size() }, broadcast_shape);
            auto broadcast_v_const = ov::op::v0::Constant::create(ov::element::i64, { broadcast_shape.size() }, broadcast_shape);
            auto broadcast_k = std::make_shared<ov::op::v3::Broadcast>(reshape1_k, broadcast_k_const, ov::op::BroadcastType::BIDIRECTIONAL);
            auto broadcast_v = std::make_shared<ov::op::v3::Broadcast>(reshape1_v, broadcast_v_const, ov::op::BroadcastType::BIDIRECTIONAL);

            // 4. Reshape back to 4D -> [1, 8, 842, 256]
            ov::Shape reshape2_shape = { q_shape[0], q_shape[1], total_seq_len, k_shape[3] };
            auto reshape2_k_const = ov::op::v0::Constant::create(ov::element::i64, { reshape2_shape.size() }, reshape2_shape);
            auto reshape2_v_const = ov::op::v0::Constant::create(ov::element::i64, { reshape2_shape.size() }, reshape2_shape);
            key_input = std::make_shared<ov::op::v1::Reshape>(broadcast_k, reshape2_k_const, true);
            value_input = std::make_shared<ov::op::v1::Reshape>(broadcast_v, reshape2_v_const, true);
        }
        else if (!is_complex_gqa && (kv_num_head_factor > 1)) {
            auto q_shape = query_shape.to_shape();                  // [1, 8, 10, 256]
            auto k_shape = key_shape.to_shape();                    // [1, 1, 10, 256]
            auto mask_shape = attention_mask_shape.to_shape();      // [1, 1]

            size_t total_seq_len = mask_shape[1];                   // 1
            size_t current_seq_len = k_shape[2];                    // 1
            size_t past_seq_len = total_seq_len - current_seq_len;  // 0

            // Static cache with full sequence length expected by updates/scatter axis.
            ov::Shape kv_cache_shape = {k_shape[0], k_shape[1], total_seq_len, k_shape[3]};
            auto past_key = std::make_shared<ov::op::v0::Parameter>(inType, kv_cache_shape);
            auto past_value = std::make_shared<ov::op::v0::Parameter>(inType, kv_cache_shape);
            model_params.push_back(past_key);
            model_params.push_back(past_value);
            // in1: [current_seq_len]
            std::vector<int64_t> update_indices(current_seq_len);
            std::iota(update_indices.begin(), update_indices.end(), static_cast<int64_t>(past_seq_len));
            auto indices =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{current_seq_len}, update_indices);

            // in3: [1], axis=2 (sequence dimension)
            auto scatter_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

            // ScatterUpdate: in0=[1,10,total_seq_len,128], in1=[current_seq_len],
            //                in2=[1,10,current_seq_len,128], in3={2}
            auto scatter_k = std::make_shared<ov::op::v3::ScatterUpdate>(past_key, indices, key_input, scatter_axis);
            auto scatter_v = std::make_shared<ov::op::v3::ScatterUpdate>(past_value, indices, value_input, scatter_axis);
            // VariadicSplit signature:
            // in0=[1,10,?,128], in1=[], in2=[3], out0/out1/out2=[1,10,?,128] (partial shapes)
            auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
            // Deterministic lengths must sum to the split axis size (k_shape[1]).
            std::vector<int64_t> split_lengths_values = {static_cast<int64_t>(k_shape[1]), int64_t{0}, int64_t{0}};
            auto split_lengths = ov::op::v0::Constant::create(ov::element::i64,
                                                               ov::Shape{3},
                                                               split_lengths_values);
            auto split_k = std::make_shared<ov::op::v1::VariadicSplit>(scatter_k, split_axis, split_lengths);
            auto split_v = std::make_shared<ov::op::v1::VariadicSplit>(scatter_v, split_axis, split_lengths);
            // Reshape 4D→5D: [1,10,?,128] -> [1,10,1,?,128]
            std::vector<int64_t> reshape1_pattern = {
                static_cast<int64_t>(k_shape[0]),
                static_cast<int64_t>(k_shape[1]),
                1,
                -1,
                static_cast<int64_t>(k_shape[3])
            };
            auto reshape1_k_const = ov::op::v0::Constant::create(ov::element::i64, {reshape1_pattern.size()}, reshape1_pattern);
            auto reshape1_v_const = ov::op::v0::Constant::create(ov::element::i64, {reshape1_pattern.size()}, reshape1_pattern);
            auto reshape1_k = std::make_shared<ov::op::v1::Reshape>(split_k->output(0), reshape1_k_const, false);
            auto reshape1_v = std::make_shared<ov::op::v1::Reshape>(split_v->output(0), reshape1_v_const, false);

            // Concat kv_num_head_factor copies to match query head expansion.
            const size_t head_expand_factor = static_cast<size_t>(kv_num_head_factor);
            ov::OutputVector concat_k_inputs(head_expand_factor, reshape1_k);
            ov::OutputVector concat_v_inputs(head_expand_factor, reshape1_v);
            auto concat_k = std::make_shared<ov::op::v0::Concat>(concat_k_inputs, 2);
            auto concat_v = std::make_shared<ov::op::v0::Concat>(concat_v_inputs, 2);
           
            // Reshape 5D->4D: [1,10,4,?,128] -> [1,40,?,128]
            std::vector<int64_t> reshape2_pattern = {
                static_cast<int64_t>(k_shape[0]),
                static_cast<int64_t>(q_shape[1]),
                -1,
                static_cast<int64_t>(k_shape[3])
            };
            auto reshape2_k_const = ov::op::v0::Constant::create(ov::element::i64, {reshape2_pattern.size()}, reshape2_pattern);
            auto reshape2_v_const = ov::op::v0::Constant::create(ov::element::i64, {reshape2_pattern.size()}, reshape2_pattern);
            key_input = std::make_shared<ov::op::v1::Reshape>(concat_k, reshape2_k_const, false);
            value_input = std::make_shared<ov::op::v1::Reshape>(concat_v, reshape2_v_const, false);
        }

        const auto mask = std::make_shared<ov::op::v0::Parameter>(inType, attention_mask_shape);
        model_params.push_back(mask);

        const auto scale_const = ov::op::v0::Constant::create(inType, {}, std::vector<float>{std::get<7>(params)});
        std::shared_ptr<ov::op::v0::MatMul> qk;
        if (reshape) {
            qk = std::make_shared<ov::op::v0::MatMul>(query_reshaped, key_input, false, true);
        } else {
            qk = std::make_shared<ov::op::v0::MatMul>(query, key_input, false, true);
        }

        const auto scaled_qk = std::make_shared<ov::op::v1::Multiply>(qk, scale_const);
        const auto mask_add = std::make_shared<ov::op::v1::Add>(scaled_qk, mask);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(mask_add, -1);
        std::shared_ptr<ov::op::v0::MatMul> qkv;
        std::shared_ptr<ov::op::v1::Reshape> qkv_reshaped;
        std::shared_ptr<ov::op::v0::Result> output;
        if (reshape) {
            qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value_input, false, false);
            const auto qkv_reshape_params =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{query_shape.size()}, query_shape.to_shape());
            qkv_reshaped = std::make_shared<ov::op::v1::Reshape>(qkv, qkv_reshape_params, true);
            output = std::make_shared<ov::op::v0::Result>(qkv_reshaped->output(0));
        } else {
            qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value_input, false, false);
            output = std::make_shared<ov::op::v0::Result>(qkv->output(0));
        }

        function = std::make_shared<ov::Model>(ov::OutputVector{output}, model_params, "sdpa_model");

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
                         ::testing::Values(std::make_tuple(ov::PartialShape{1, 40, 1, 128},
                                                            ov::Shape{1, 40, 1, 128},
                                                            ov::PartialShape{1, 10, 1, 128},
                                                            ov::Shape{1, 10, 1, 128},
                                                            ov::PartialShape{1, 10, 1, 128},
                                                            ov::Shape{1, 10, 1, 128},
                                                            ov::PartialShape{1, 1},
                                                            1.0f,
                                                            0.025f,
                                                            0.025f,
                                                            false,
                                                            4),
                                        std::make_tuple(ov::PartialShape{10, 1024, 64},
                                                          ov::Shape{10, 1024, 64},
                                                          ov::PartialShape{10, 77, 64},
                                                          ov::Shape{10, 77, 64},
                                                          ov::PartialShape{10, 77, 64},
                                                          ov::Shape{10, 77, 64},
                                                          ov::PartialShape{1024, 77},
                                                          1.0f,
                                                          0.025f,
                                                          0.025f,
                                                          false,
                                                          1),
                                            std::make_tuple(ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{10, 1024, 77},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f,
                                                           false,
                                                           1),
                                           std::make_tuple(ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{10, 1024, 64},
                                                           ov::PartialShape{10, 1024, 1024},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f,
                                                           false,
                                                           1),
                                           std::make_tuple(ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{1, 10, 77, 64},
                                                           ov::Shape{10, 77, 64},
                                                           ov::PartialShape{77, 77},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f,
                                                           false,
                                                           1),
                                           std::make_tuple(ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{1, 10, 1024, 64},
                                                           ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{1, 10, 1024, 64},
                                                           ov::PartialShape{1, 10, 1024, 64},
                                                           ov::Shape{1, 10, 1024, 64},
                                                           ov::PartialShape{10, 1024, 1024},
                                                           1.0f,
                                                           0.025f,
-                                                          0.025f,
                                                           false,
                                                           1),
                                           std::make_tuple(ov::PartialShape{1, 8, 10, 256},
                                                           ov::Shape{1, 8, 10, 256},
                                                           ov::PartialShape{1, 1, 10, 256},
                                                           ov::Shape{1, 1, 10, 256},
                                                           ov::PartialShape{1, 1, 10, 256},
                                                           ov::Shape{1, 1, 10, 256},
                                                           ov::PartialShape{10, 842},
                                                           1.0f,
                                                           0.025f,
                                                           0.025f,
                                                           true,
                                                           8)));

}  // namespace
