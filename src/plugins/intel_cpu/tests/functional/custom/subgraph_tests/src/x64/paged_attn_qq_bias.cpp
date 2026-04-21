// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {

struct QQBiasPattern {
    std::string name;
    std::vector<uint8_t> matrix;  // N×N matrix for N draft tokens
    int32_t num_draft_tokens;
};

using PagedAttnQQBiasParams = std::tuple<ElementType, size_t, size_t, size_t, QQBiasPattern>;

class PagedAttnQQBiasTest : public testing::WithParamInterface<PagedAttnQQBiasParams>,
                            virtual public ov::test::SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnQQBiasParams>& obj) {
        const auto& [inType, past_len, num_draft, head_size, pattern] = obj.param;
        std::ostringstream result;
        result << "Prc=" << inType << "_";
        result << "PastLen=" << past_len << "_";
        result << "NumDraft=" << num_draft << "_";
        result << "HS=" << head_size << "_";
        result << "Pattern=" << pattern.name;
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

    std::shared_ptr<ov::Model> get_pa_model(ov::element::Type data_type,
                                            ov::Dimension::value_type head_size,
                                            ov::Dimension::value_type head_num,
                                            bool enable_qq_bias) {
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, data_type, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");
        auto key_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                    ov::element::dynamic, "key_cache.0");
        auto value_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                      ov::element::dynamic, "value_cache.0");
        auto past_lens = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
        auto subsequence_begins = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
        auto block_indices = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
        auto block_indices_begins = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");

        float scale_value = 1.0f / std::sqrt(static_cast<float>(head_size));
        auto scale = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_value});
        auto sliding_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto alibi_slopes = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto max_context_len = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{1024});
        auto score_aggregation_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto rotated_block_indices = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto rotation_deltas = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto rotation_trig_lut = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
        auto xattention_threshold = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
        auto xattention_block_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{64});
        auto xattention_stride = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{8});
        auto sinks = std::static_pointer_cast<v0::Constant>(
            ov::test::utils::make_constant(data_type, Shape{0}));
        auto adaptive_rkv_start_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto adaptive_rkv_evictable_sizes = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto adaptive_rkv_diversity_block_set_indices = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto adaptive_rkv_diversity_block_set_indices_begins = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto token_type_ids = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});

        ParameterVector params = {q, k, v, key_cache, value_cache, past_lens,
                                  subsequence_begins, block_indices, block_indices_begins};

        OutputVector pa_inputs = {q, k, v, key_cache, value_cache,
                                  past_lens, subsequence_begins, block_indices, block_indices_begins,
                                  scale, sliding_window, alibi_slopes, max_context_len,
                                  score_aggregation_window, rotated_block_indices, rotation_deltas,
                                  rotation_trig_lut, xattention_threshold, xattention_block_size,
                                  xattention_stride, sinks, adaptive_rkv_start_size,
                                  adaptive_rkv_evictable_sizes,
                                  adaptive_rkv_diversity_block_set_indices,
                                  adaptive_rkv_diversity_block_set_indices_begins,
                                  token_type_ids};

        // Create qq_bias and qq_bias_begins as Parameters when enabled
        if (enable_qq_bias) {
            auto qq_bias_param = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::u8, "qq_bias");
            auto qq_bias_begins_param = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "qq_bias_begins");
            params.push_back(qq_bias_param);
            params.push_back(qq_bias_begins_param);
            pa_inputs.push_back(qq_bias_param);
            pa_inputs.push_back(qq_bias_begins_param);
        } else {
            // Use empty Constants when disabled
            auto qq_bias = std::make_shared<v0::Constant>(ov::element::u8, Shape{0}, std::vector<uint8_t>{});
            auto qq_bias_begins = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{});
            pa_inputs.push_back(qq_bias);
            pa_inputs.push_back(qq_bias_begins);
        }

        OPENVINO_ASSERT(pa_inputs.size() == 28);

        auto paged_attn = std::make_shared<op::PagedAttentionExtension>(pa_inputs);
        paged_attn->get_rt_info()["num_k_heads"] = head_num;
        paged_attn->get_rt_info()["k_head_size"] = head_size;
        paged_attn->get_rt_info()["num_v_heads"] = head_num;
        paged_attn->get_rt_info()["v_head_size"] = head_size;

        return std::make_shared<ov::Model>(OutputVector{paged_attn}, params);
    }

    struct RunResult {
        ov::Tensor prefill_output;
        ov::Tensor decode_output;
        ov::Tensor key_cache;
        ov::Tensor value_cache;
    };

    // Run two iterations: prefill + decode
    // If enable_qq_bias is true, decode phase uses provided qq_bias; otherwise standard causal
    RunResult run_pa(std::shared_ptr<ov::Model> model,
                    ov::element::Type data_type,
                    size_t prefill_len,
                    size_t num_draft,
                    size_t head_size,
                    size_t head_num,
                    bool enable_qq_bias,
                    const std::vector<uint8_t>& qq_bias_data = {},
                    const std::vector<int32_t>& qq_bias_begins_data = {}) {
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        function = model;
        compile_model();
        auto infer_request = compiledModel.create_infer_request();

        auto params = model->get_parameters();
        size_t hidden_dim = head_num * head_size;
        size_t batch_size = 1;

        // Helper to fill tensors with test data
        auto fill_tensor = [](ov::Tensor& t, float base, float stride) {
            if (t.get_element_type() == ov::element::f32) {
                auto* p = t.data<float>();
                for (size_t i = 0; i < t.get_size(); i++) {
                    p[i] = base + stride * static_cast<float>(i % 17);
                }
            } else if (t.get_element_type() == ov::element::f16) {
                auto* p = t.data<ov::float16>();
                for (size_t i = 0; i < t.get_size(); i++) {
                    p[i] = ov::float16(base + stride * static_cast<float>(i % 17));
                }
            }
        };

        // Determine cache precision and create KV cache tensors
        ov::Tensor key_cache_tensor, value_cache_tensor;
        for (const auto& input : compiledModel.inputs()) {
            for (auto& name : input.get_names()) {
                if (name.find("key_cache.") == 0) {
                    auto cache_precision = input.get_element_type();
                    auto pshape = input.get_partial_shape();
                    size_t total_tokens = prefill_len + num_draft;
                    size_t block_nums = (total_tokens + 31) / 32;
                    pshape[0] = block_nums;
                    key_cache_tensor = ov::Tensor(cache_precision, pshape.get_shape());
                } else if (name.find("value_cache.") == 0) {
                    auto cache_precision = input.get_element_type();
                    auto pshape = input.get_partial_shape();
                    size_t total_tokens = prefill_len + num_draft;
                    size_t block_nums = (total_tokens + 31) / 32;
                    pshape[0] = block_nums;
                    value_cache_tensor = ov::Tensor(cache_precision, pshape.get_shape());
                }
            }
        }

        // ========== Iteration 1: Prefill phase ==========
        {
            ov::Tensor q_prefill(data_type, {prefill_len, hidden_dim});
            ov::Tensor k_prefill(data_type, {prefill_len, hidden_dim});
            ov::Tensor v_prefill(data_type, {prefill_len, hidden_dim});

            fill_tensor(q_prefill, 0.1f, 0.01f);
            fill_tensor(k_prefill, 0.2f, 0.01f);
            fill_tensor(v_prefill, 0.3f, 0.01f);

            size_t prefill_blocks = (prefill_len + 31) / 32;
            ov::Tensor past_lens(ov::element::i32, {batch_size});
            ov::Tensor subsequence_begins(ov::element::i32, {batch_size + 1});
            ov::Tensor block_indices(ov::element::i32, {prefill_blocks});
            ov::Tensor block_indices_begins(ov::element::i32, {batch_size + 1});

            past_lens.data<int32_t>()[0] = 0;
            subsequence_begins.data<int32_t>()[0] = 0;
            subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(prefill_len);
            block_indices_begins.data<int32_t>()[0] = 0;
            block_indices_begins.data<int32_t>()[1] = static_cast<int32_t>(prefill_blocks);
            for (size_t i = 0; i < prefill_blocks; i++) {
                block_indices.data<int32_t>()[i] = static_cast<int32_t>(i);
            }

            // Empty qq_bias for prefill
            ov::Tensor qq_bias_empty(ov::element::u8, {0});
            ov::Tensor qq_bias_begins_empty(ov::element::i32, {batch_size + 1});
            for (size_t i = 0; i <= batch_size; i++) {
                qq_bias_begins_empty.data<int32_t>()[i] = 0;
            }

            for (auto& param : params) {
                auto name = param->get_friendly_name();
                if (name == "q") infer_request.set_tensor(param, q_prefill);
                else if (name == "k") infer_request.set_tensor(param, k_prefill);
                else if (name == "v") infer_request.set_tensor(param, v_prefill);
                else if (name == "key_cache.0") infer_request.set_tensor(param, key_cache_tensor);
                else if (name == "value_cache.0") infer_request.set_tensor(param, value_cache_tensor);
                else if (name == "past_lens") infer_request.set_tensor(param, past_lens);
                else if (name == "subsequence_begins") infer_request.set_tensor(param, subsequence_begins);
                else if (name == "block_indices") infer_request.set_tensor(param, block_indices);
                else if (name == "block_indices_begins") infer_request.set_tensor(param, block_indices_begins);
                else if (name == "qq_bias") infer_request.set_tensor(param, qq_bias_empty);
                else if (name == "qq_bias_begins") infer_request.set_tensor(param, qq_bias_begins_empty);
            }

            infer_request.infer();
        }

        // ========== Iteration 2: Decode phase ==========
        ov::Tensor decode_output;
        {
            ov::Tensor q_decode(data_type, {num_draft, hidden_dim});
            ov::Tensor k_decode(data_type, {num_draft, hidden_dim});
            ov::Tensor v_decode(data_type, {num_draft, hidden_dim});

            fill_tensor(q_decode, 0.4f, 0.02f);
            fill_tensor(k_decode, 0.5f, 0.02f);
            fill_tensor(v_decode, 0.6f, 0.02f);

            size_t total_tokens = prefill_len + num_draft;
            size_t total_blocks = (total_tokens + 31) / 32;
            ov::Tensor past_lens(ov::element::i32, {batch_size});
            ov::Tensor subsequence_begins(ov::element::i32, {batch_size + 1});
            ov::Tensor block_indices(ov::element::i32, {total_blocks});
            ov::Tensor block_indices_begins(ov::element::i32, {batch_size + 1});

            past_lens.data<int32_t>()[0] = static_cast<int32_t>(prefill_len);
            subsequence_begins.data<int32_t>()[0] = 0;
            subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(num_draft);
            block_indices_begins.data<int32_t>()[0] = 0;
            block_indices_begins.data<int32_t>()[1] = static_cast<int32_t>(total_blocks);
            for (size_t i = 0; i < total_blocks; i++) {
                block_indices.data<int32_t>()[i] = static_cast<int32_t>(i);
            }

            // Prepare qq_bias tensors based on enable_qq_bias flag
            ov::Tensor qq_bias_decode, qq_bias_begins_decode;
            if (enable_qq_bias && !qq_bias_data.empty()) {
                // Use provided qq_bias data
                qq_bias_decode = ov::Tensor(ov::element::u8, {qq_bias_data.size()});
                std::memcpy(qq_bias_decode.data<uint8_t>(), qq_bias_data.data(), qq_bias_data.size());

                qq_bias_begins_decode = ov::Tensor(ov::element::i32, {qq_bias_begins_data.size()});
                std::memcpy(qq_bias_begins_decode.data<int32_t>(), qq_bias_begins_data.data(),
                           qq_bias_begins_data.size() * sizeof(int32_t));
            } else {
                // No qq_bias (standard causal attention) - model without qq_bias params won't need these
                // These won't be set if model was created with enable_qq_bias=false
            }

            for (auto& param : params) {
                auto name = param->get_friendly_name();
                if (name == "q") infer_request.set_tensor(param, q_decode);
                else if (name == "k") infer_request.set_tensor(param, k_decode);
                else if (name == "v") infer_request.set_tensor(param, v_decode);
                else if (name == "key_cache.0") infer_request.set_tensor(param, key_cache_tensor);
                else if (name == "value_cache.0") infer_request.set_tensor(param, value_cache_tensor);
                else if (name == "past_lens") infer_request.set_tensor(param, past_lens);
                else if (name == "subsequence_begins") infer_request.set_tensor(param, subsequence_begins);
                else if (name == "block_indices") infer_request.set_tensor(param, block_indices);
                else if (name == "block_indices_begins") infer_request.set_tensor(param, block_indices_begins);
                else if (name == "qq_bias" && enable_qq_bias) infer_request.set_tensor(param, qq_bias_decode);
                else if (name == "qq_bias_begins" && enable_qq_bias) infer_request.set_tensor(param, qq_bias_begins_decode);
            }

            infer_request.infer();

            auto output = infer_request.get_output_tensor(0);
            decode_output = ov::Tensor{output.get_element_type(), output.get_shape()};
            output.copy_to(decode_output);
        }

        return {ov::Tensor{}, decode_output, key_cache_tensor, value_cache_tensor};
    }

};

// Test that qq_bias tree mask produces different output than full causal mask
TEST_P(PagedAttnQQBiasTest, SpeculativeDecodingTreeMask) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, prefill_len, num_draft, head_size, pattern] = this->GetParam();

    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    targetDevice = ov::test::utils::DEVICE_CPU;

    size_t head_num = 8;

    // Run with tree mask (qq_bias enabled with pattern)
    std::vector<int32_t> qq_bias_begins_tree = {0, static_cast<int32_t>(pattern.matrix.size())};
    auto model_tree = get_pa_model(inType, head_size, head_num, true);
    auto result_tree = run_pa(model_tree, inType, prefill_len, num_draft,
                             head_size, head_num, true,
                             pattern.matrix, qq_bias_begins_tree);

    // Run reference without qq_bias (standard causal attention)
    auto model_ref = get_pa_model(inType, head_size, head_num, false);
    auto result_ref = run_pa(model_ref, inType, prefill_len, num_draft,
                            head_size, head_num, false);

    // Validate outputs
    ASSERT_GT(result_tree.decode_output.get_size(), 0);
    ASSERT_GT(result_ref.decode_output.get_size(), 0);
    ASSERT_EQ(result_tree.decode_output.get_size(), result_ref.decode_output.get_size());

    // Check for NaN
    auto check_no_nan = [](const ov::Tensor& t, const std::string& name) {
        if (t.get_element_type() == ov::element::f32) {
            auto* data = t.data<float>();
            for (size_t i = 0; i < t.get_size(); i++) {
                EXPECT_FALSE(std::isnan(data[i])) << name << " contains NaN at index " << i;
            }
        } else if (t.get_element_type() == ov::element::f16) {
            auto* data = t.data<ov::float16>();
            for (size_t i = 0; i < t.get_size(); i++) {
                float val = static_cast<float>(data[i]);
                EXPECT_FALSE(std::isnan(val)) << name << " contains NaN at index " << i;
            }
        }
    };

    check_no_nan(result_tree.decode_output, "Tree mask output");
    check_no_nan(result_ref.decode_output, "Reference output");

    // For tree patterns that differ from standard causal, outputs should differ
    // Standard causal mask for N tokens: q_i can attend to k_0..k_i
    std::vector<uint8_t> standard_causal_mask;
    int N = pattern.num_draft_tokens;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            standard_causal_mask.push_back((j <= i) ? 1 : 0);
        }
    }

    bool pattern_differs_from_causal = false;
    for (size_t i = 0; i < pattern.matrix.size(); i++) {
        if (pattern.matrix[i] != standard_causal_mask[i]) {
            pattern_differs_from_causal = true;
            break;
        }
    }

        // Compare outputs - they should differ when patterns differ
        float max_diff = 0.0f;
        size_t diff_count = 0;

        if (inType == ov::element::f32) {
            auto* data_tree = result_tree.decode_output.data<float>();
            auto* data_ref = result_ref.decode_output.data<float>();
            for (size_t i = 0; i < result_tree.decode_output.get_size(); i++) {
                float diff = std::abs(data_tree[i] - data_ref[i]);
                if (diff > max_diff) max_diff = diff;
                if (diff > 1e-5f) diff_count++;
            }
        } else if (inType == ov::element::f16) {
            auto* data_tree = result_tree.decode_output.data<ov::float16>();
            auto* data_ref = result_ref.decode_output.data<ov::float16>();
            for (size_t i = 0; i < result_tree.decode_output.get_size(); i++) {
                float diff = std::abs(static_cast<float>(data_tree[i]) - static_cast<float>(data_ref[i]));
                if (diff > max_diff) max_diff = diff;
                if (diff > 1e-3f) diff_count++;
            }
        }

        // Outputs should differ when tree pattern != full causal
        // Note: Due to test data characteristics, differences may be subtle
        // We primarily verify that the mechanism executes correctly without errors
        if (pattern_differs_from_causal)
            EXPECT_TRUE(diff_count > 0) << "expect mask effective for tree attention ";
        else
            EXPECT_TRUE(diff_count == 0) << "expect same result for full attention";
}

// Define test patterns for speculative decoding
const std::vector<QQBiasPattern> qq_bias_patterns = {
    {
        "tree_4tokens",
        // 4×4 matrix for 4 draft tokens with tree structure:
        // q0: [1 0 0 0] - only sees k0
        // q1: [1 1 0 0] - sees k0, k1
        // q2: [1 0 1 0] - sees k0, k2 (skips k1)
        // q3: [1 0 1 1] - sees k0, k2, k3 (skips k1)
        {1, 0, 0, 0,
         1, 1, 0, 0,
         1, 0, 1, 0,
         1, 0, 1, 1},
        4
    },
    {
        "diagonal_4tokens",
        // 4×4 diagonal (standard causal)
        {1, 0, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 0,
         1, 1, 1, 1},
        4
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnQQBias,
                         PagedAttnQQBiasTest,
                         ::testing::Combine(
                             ::testing::Values(ElementType::f32, ElementType::f16),
                             ::testing::Values(4, 8),      // past_len
                             ::testing::Values(4),          // num_draft tokens
                             ::testing::Values(64),         // head_size
                             ::testing::ValuesIn(qq_bias_patterns)),
                         PagedAttnQQBiasTest::getTestCaseName);

}  // namespace test
}  // namespace ov
