// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::test;
using namespace ov::op;

namespace ov {
namespace test {

struct TokenTypePattern {
    std::string name;
    std::vector<int32_t> types;  // 0=text, 1=image
};

using PagedAttnTokenTypeParams = std::tuple<ElementType, size_t, size_t, TokenTypePattern>;

class PagedAttnTokenTypeTestGPU : public testing::WithParamInterface<PagedAttnTokenTypeParams>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnTokenTypeParams>& obj) {
        const auto& [inType, head_size, head_num, pattern] = obj.param;
        std::ostringstream result;
        result << "Prc=" << inType << "_";
        result << "HS=" << head_size << "_";
        result << "HN=" << head_num << "_";
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
                                            ov::Dimension::value_type head_num) {
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");
        // GPU plugin expects 4-dim cache with concrete element type
        // key_cache: [num_blocks, num_kv_heads, head_size, block_size]
        // value_cache: [num_blocks, num_kv_heads, block_size, head_size]
        const int64_t block_size = 16;
        auto key_cache = make_param(PartialShape{ov::Dimension::dynamic(), head_num, head_size, block_size},
                                    data_type, "key_cache.0");
        auto value_cache = make_param(PartialShape{ov::Dimension::dynamic(), head_num, block_size, head_size},
                                      data_type, "value_cache.0");
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

        auto token_type_ids = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "token_type_ids");

        ParameterVector params = {q, k, v, key_cache, value_cache, past_lens,
                                  subsequence_begins, block_indices, block_indices_begins,
                                  token_type_ids};

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

        OPENVINO_ASSERT(pa_inputs.size() == 26);

        auto paged_attn = std::make_shared<op::PagedAttentionExtension>(pa_inputs);
        paged_attn->get_rt_info()["num_k_heads"] = head_num;
        paged_attn->get_rt_info()["k_head_size"] = head_size;
        paged_attn->get_rt_info()["num_v_heads"] = head_num;
        paged_attn->get_rt_info()["v_head_size"] = head_size;

        return std::make_shared<ov::Model>(OutputVector{paged_attn}, params);
    }

    struct RunResult {
        ov::Tensor output;
    };

    RunResult run_pa_with_token_types(std::shared_ptr<ov::Model> model,
                                     ov::element::Type data_type,
                                     size_t seq_len,
                                     size_t head_size,
                                     size_t head_num,
                                     const std::vector<int32_t>& token_types) {
        OPENVINO_ASSERT(token_types.size() == seq_len);

        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        configuration[ov::hint::kv_cache_precision.name()] = data_type;
        function = model;
        compile_model();
        auto infer_request = compiledModel.create_infer_request();

        // Create cache tensors with known shapes
        const size_t block_size = 16;
        const size_t block_nums = 1024 / block_size;
        ov::Tensor key_cache_tensor(data_type, {block_nums, head_num, head_size, block_size});
        ov::Tensor value_cache_tensor(data_type, {block_nums, head_num, block_size, head_size});

        auto params = model->get_parameters();
        size_t hidden_dim = head_num * head_size;

        // q, k, v tensors [seq_len, hidden_dim]
        ov::Tensor q_tensor(data_type, {seq_len, hidden_dim});
        ov::Tensor k_tensor(data_type, {seq_len, hidden_dim});
        ov::Tensor v_tensor(data_type, {seq_len, hidden_dim});

        if (data_type == ov::element::f32) {
            auto fill_f32 = [](ov::Tensor& t, float base, float stride) {
                auto* p = t.data<float>();
                for (size_t i = 0; i < t.get_size(); i++) {
                    p[i] = base + stride * static_cast<float>(i % 17);
                }
            };
            fill_f32(q_tensor, 0.1f, 0.01f);
            fill_f32(k_tensor, 0.2f, 0.01f);
            fill_f32(v_tensor, 0.3f, 0.01f);
        } else if (data_type == ov::element::f16) {
            auto fill_f16 = [](ov::Tensor& t, float base, float stride) {
                auto* p = t.data<ov::float16>();
                for (size_t i = 0; i < t.get_size(); i++) {
                    p[i] = ov::float16(base + stride * static_cast<float>(i % 17));
                }
            };
            fill_f16(q_tensor, 0.1f, 0.01f);
            fill_f16(k_tensor, 0.2f, 0.01f);
            fill_f16(v_tensor, 0.3f, 0.01f);
        }

        // Prefill: past_lens=0, single sequence
        size_t batch_size = 1;
        int32_t total_blocks = static_cast<int32_t>((seq_len + block_size - 1) / block_size);

        ov::Tensor past_lens(ov::element::i32, {batch_size});
        ov::Tensor subsequence_begins(ov::element::i32, {batch_size + 1});
        ov::Tensor block_indices(ov::element::i32, {static_cast<size_t>(total_blocks)});
        ov::Tensor block_indices_begins(ov::element::i32, {batch_size + 1});

        past_lens.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(seq_len);
        block_indices_begins.data<int32_t>()[0] = 0;
        block_indices_begins.data<int32_t>()[1] = total_blocks;
        for (int32_t i = 0; i < total_blocks; i++) {
            block_indices.data<int32_t>()[i] = i;
        }

        // token_type_ids
        ov::Tensor token_type_tensor(ov::element::i32, {seq_len});
        std::memcpy(token_type_tensor.data<int32_t>(), token_types.data(), seq_len * sizeof(int32_t));

        for (auto& param : params) {
            auto name = param->get_friendly_name();
            if (name == "q") infer_request.set_tensor(param, q_tensor);
            else if (name == "k") infer_request.set_tensor(param, k_tensor);
            else if (name == "v") infer_request.set_tensor(param, v_tensor);
            else if (name == "key_cache.0") infer_request.set_tensor(param, key_cache_tensor);
            else if (name == "value_cache.0") infer_request.set_tensor(param, value_cache_tensor);
            else if (name == "past_lens") infer_request.set_tensor(param, past_lens);
            else if (name == "subsequence_begins") infer_request.set_tensor(param, subsequence_begins);
            else if (name == "block_indices") infer_request.set_tensor(param, block_indices);
            else if (name == "block_indices_begins") infer_request.set_tensor(param, block_indices_begins);
            else if (name == "token_type_ids") infer_request.set_tensor(param, token_type_tensor);
        }

        infer_request.infer();

        auto output = infer_request.get_output_tensor(0);
        ov::Tensor output_copy{output.get_element_type(), output.get_shape()};
        output.copy_to(output_copy);
        return {output_copy};
    }

    // Helper to get output data as float, handling f16 conversion
    std::vector<float> get_output_as_float(const ov::Tensor& tensor) {
        std::vector<float> result(tensor.get_size());
        if (tensor.get_element_type() == ov::element::f32) {
            auto* p = tensor.data<float>();
            std::copy(p, p + tensor.get_size(), result.begin());
        } else if (tensor.get_element_type() == ov::element::f16) {
            auto* p = tensor.data<ov::float16>();
            for (size_t i = 0; i < tensor.get_size(); i++) {
                result[i] = static_cast<float>(p[i]);
            }
        }
        return result;
    }
};

// With all-zero token_type_ids (text-only), causal masking must hold:
// output at position i depends only on tokens 0..i, so a shorter prefix
// should produce identical outputs for the shared positions.
TEST_P(PagedAttnTokenTypeTestGPU, AllTextIsCausal) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();

    targetDevice = ov::test::utils::DEVICE_GPU;

    size_t full_len = pattern.types.size();
    size_t prefix_len = full_len / 2;
    OPENVINO_ASSERT(prefix_len > 0);

    std::vector<int32_t> all_text_full(full_len, 0);
    std::vector<int32_t> all_text_prefix(prefix_len, 0);

    auto model_full = get_pa_model(inType, head_size, head_num);
    auto result_full = run_pa_with_token_types(model_full, inType, full_len, head_size, head_num, all_text_full);

    auto model_prefix = get_pa_model(inType, head_size, head_num);
    auto result_prefix = run_pa_with_token_types(model_prefix, inType, prefix_len, head_size, head_num, all_text_prefix);

    size_t hidden_dim = head_num * head_size;
    auto full_data = get_output_as_float(result_full.output);
    auto prefix_data = get_output_as_float(result_prefix.output);

    const float tolerance = (inType == ElementType::f16) ? 1e-2f : 1e-5f;

    for (size_t pos = 0; pos < prefix_len; pos++) {
        for (size_t d = 0; d < hidden_dim; d++) {
            float diff = std::abs(full_data[pos * hidden_dim + d] - prefix_data[pos * hidden_dim + d]);
            EXPECT_LT(diff, tolerance)
                << "Causal masking violated: position " << pos << " dim " << d
                << " differs between seq_len=" << full_len << " and seq_len=" << prefix_len
                << ", diff=" << diff;
        }
    }
}

TEST_P(PagedAttnTokenTypeTestGPU, ImageTokensDifferFromCausal) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();

    targetDevice = ov::test::utils::DEVICE_GPU;

    size_t seq_len = pattern.types.size();

    auto pa_model = get_pa_model(inType, head_size, head_num);
    auto result_bidir = run_pa_with_token_types(pa_model, inType, seq_len, head_size, head_num, pattern.types);

    std::vector<int32_t> all_causal(seq_len, 0);
    auto pa_model_causal = get_pa_model(inType, head_size, head_num);
    auto result_causal = run_pa_with_token_types(pa_model_causal, inType, seq_len, head_size, head_num, all_causal);

    size_t hidden_dim = head_num * head_size;
    auto bidir_data = get_output_as_float(result_bidir.output);
    auto causal_data = get_output_as_float(result_causal.output);

    const float tolerance = (inType == ElementType::f16) ? 1e-2f : 1e-5f;

    bool any_image_differs = false;
    for (size_t pos = 0; pos < seq_len; pos++) {
        if (pattern.types[pos] != 1) continue;
        for (size_t d = 0; d < hidden_dim; d++) {
            float diff = std::abs(bidir_data[pos * hidden_dim + d] - causal_data[pos * hidden_dim + d]);
            if (diff > tolerance) {
                any_image_differs = true;
                break;
            }
        }
        if (any_image_differs) break;
    }
    EXPECT_TRUE(any_image_differs)
        << "Expected image tokens with bidirectional attention to produce different output than causal-only";
}

// Text tokens outside image groups should be unaffected by token_type_ids
TEST_P(PagedAttnTokenTypeTestGPU, TextTokensUnaffected) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();

    targetDevice = ov::test::utils::DEVICE_GPU;

    size_t seq_len = pattern.types.size();

    auto pa_model = get_pa_model(inType, head_size, head_num);
    auto result_bidir = run_pa_with_token_types(pa_model, inType, seq_len, head_size, head_num, pattern.types);

    std::vector<int32_t> all_causal(seq_len, 0);
    auto pa_model_causal = get_pa_model(inType, head_size, head_num);
    auto result_causal = run_pa_with_token_types(pa_model_causal, inType, seq_len, head_size, head_num, all_causal);

    size_t hidden_dim = head_num * head_size;
    auto bidir_data = get_output_as_float(result_bidir.output);
    auto causal_data = get_output_as_float(result_causal.output);

    const float tolerance = (inType == ElementType::f16) ? 1e-2f : 1e-5f;

    size_t first_image_pos = seq_len;
    for (size_t i = 0; i < seq_len; i++) {
        if (pattern.types[i] == 1) { first_image_pos = i; break; }
    }

    for (size_t pos = 0; pos < first_image_pos; pos++) {
        for (size_t d = 0; d < hidden_dim; d++) {
            float diff = std::abs(bidir_data[pos * hidden_dim + d] - causal_data[pos * hidden_dim + d]);
            EXPECT_LT(diff, tolerance)
                << "Text token at position " << pos << " dim " << d
                << " should be unaffected by token_type_ids, but diff=" << diff;
        }
    }
}

// Text tokens after the last image group should match causal baseline
TEST_P(PagedAttnTokenTypeTestGPU, PostImageTextIsCausal) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();

    targetDevice = ov::test::utils::DEVICE_GPU;

    size_t seq_len = pattern.types.size();

    auto pa_model = get_pa_model(inType, head_size, head_num);
    auto result_bidir = run_pa_with_token_types(pa_model, inType, seq_len, head_size, head_num, pattern.types);

    std::vector<int32_t> all_causal(seq_len, 0);
    auto pa_model_causal = get_pa_model(inType, head_size, head_num);
    auto result_causal = run_pa_with_token_types(pa_model_causal, inType, seq_len, head_size, head_num, all_causal);

    size_t hidden_dim = head_num * head_size;
    auto bidir_data = get_output_as_float(result_bidir.output);
    auto causal_data = get_output_as_float(result_causal.output);

    const float tolerance = (inType == ElementType::f16) ? 1e-2f : 1e-5f;

    size_t last_image_pos = 0;
    for (size_t i = 0; i < seq_len; i++) {
        if (pattern.types[i] == 1) last_image_pos = i;
    }

    for (size_t pos = last_image_pos + 1; pos < seq_len; pos++) {
        ASSERT_EQ(pattern.types[pos], 0) << "Expected text token at position " << pos;
        for (size_t d = 0; d < hidden_dim; d++) {
            float diff = std::abs(bidir_data[pos * hidden_dim + d] - causal_data[pos * hidden_dim + d]);
            EXPECT_LT(diff, tolerance)
                << "Post-image text token at position " << pos << " dim " << d
                << " should match causal baseline, but diff=" << diff;
        }
    }
}

namespace {

const std::vector<TokenTypePattern> token_type_patterns = {
    {"centered_image", {0, 0, 0, 1, 1, 1, 1, 0, 0, 0}},
    {"early_image", {0, 1, 1, 1, 1, 0, 0, 0, 0, 0}},
    {"late_image", {0, 0, 0, 0, 0, 1, 1, 1, 1, 0}},
    {"large_image", {0, 1, 1, 1, 1, 1, 1, 1, 1, 0}},
    {"two_image_groups", {0, 1, 1, 0, 0, 0, 1, 1, 1, 0}},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnTokenType,
                         PagedAttnTokenTypeTestGPU,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::Values(64),   // head_size
                                            ::testing::Values(8),    // head_num
                                            ::testing::ValuesIn(token_type_patterns)),
                         PagedAttnTokenTypeTestGPU::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
