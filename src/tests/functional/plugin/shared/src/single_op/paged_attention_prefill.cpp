// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_attention_prefill.hpp"

#include <cmath>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/reference/utils/paged_cache_manager_helper.hpp"

namespace ov {
namespace test {
namespace paged_attention_prefill {
namespace {

constexpr size_t GPU_BLOCK_SIZE = 16;
constexpr size_t CPU_BLOCK_SIZE = 32;

size_t block_size(const ModelOptions& options) {
    return options.cpu_layout ? CPU_BLOCK_SIZE : GPU_BLOCK_SIZE;
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant> constant(ov::element::Type type, const ov::Shape& shape, T value) {
    return ov::op::v0::Constant::create(type, shape, std::vector<T>(ov::shape_size(shape), value));
}

}  // namespace

std::shared_ptr<ov::Model> make_model(const ModelOptions& options) {
    using ov::test::utils::make_param;
    const auto type = options.data_type;
    const auto head_size = options.head_size;
    const auto head_num = options.head_num;
    const auto block = static_cast<int64_t>(block_size(options));

    auto q = make_param(type, PartialShape{-1, head_num * head_size}, "q");
    auto k = make_param(type, PartialShape{-1, head_num * head_size}, "k");
    auto v = make_param(type, PartialShape{-1, head_num * head_size}, "v");
    // GPU: key_cache [blocks, kv_heads, head_size, block], value_cache [blocks, kv_heads, block, head_size].
    // CPU: both [blocks, kv_heads, block, head_size].
    auto key_cache = make_param(type,
                                options.cpu_layout ? PartialShape{-1, head_num, block, head_size}
                                                   : PartialShape{-1, head_num, head_size, block},
                                "key_cache.0");
    auto value_cache = make_param(type, PartialShape{-1, head_num, block, head_size}, "value_cache.0");
    ParameterVector params{q,
                           k,
                           v,
                           key_cache,
                           value_cache,
                           make_param(ov::element::i32, PartialShape{-1}, "past_lens"),
                           make_param(ov::element::i32, PartialShape{-1}, "subsequence_begins"),
                           make_param(ov::element::i32, PartialShape{-1}, "block_indices"),
                           make_param(ov::element::i32, PartialShape{-1}, "block_indices_begins")};

    const auto i32 = ov::element::i32;
    const auto f32 = ov::element::f32;
    OutputVector pa_inputs(params.begin(), params.end());
    pa_inputs.push_back(constant(f32, {}, 1.0f / std::sqrt(static_cast<float>(head_size))));  // scale
    pa_inputs.push_back(constant(i32, {}, options.sliding_window));
    pa_inputs.push_back(constant(f32, {0}, 0.f));                                   // alibi_slopes
    pa_inputs.push_back(constant(i32, {}, static_cast<int32_t>(MAX_CONTEXT_LEN)));  // max_context_len
    // CPU expects a per-sequence score aggregation window; zeros disable it.
    pa_inputs.push_back(options.cpu_layout ? constant(i32, {options.batch_size}, 0) : constant(i32, {}, 0));
    pa_inputs.push_back(constant(i32, {0}, 0));                           // rotated_block_indices
    pa_inputs.push_back(constant(i32, {0}, 0));                           // rotation_deltas
    pa_inputs.push_back(constant(f32, {0}, 0.f));                         // rotation_trig_lut
    pa_inputs.push_back(constant(f32, {0}, 0.f));                         // xattention_threshold
    pa_inputs.push_back(constant(i32, {}, 0));                            // xattention_block_size
    pa_inputs.push_back(constant(i32, {}, 0));                            // xattention_stride
    pa_inputs.push_back(ov::test::utils::make_constant(type, Shape{0}));  // sinks
    pa_inputs.push_back(constant(i32, {}, 0));                            // adaptive_rkv_start_size
    pa_inputs.push_back(constant(i32, {0}, 0));                           // adaptive_rkv_evictable_sizes
    pa_inputs.push_back(constant(i32, {0}, 0));                           // adaptive_rkv_diversity_block_set_indices
    pa_inputs.push_back(constant(i32, {0}, 0));  // adaptive_rkv_diversity_block_set_indices_begins
    if (options.token_type_ids) {
        params.push_back(make_param(i32, PartialShape{-1}, "token_type_ids"));
        pa_inputs.push_back(params.back());
    } else {
        pa_inputs.push_back(constant(i32, {0}, 0));
    }
    pa_inputs.push_back(constant(ov::element::u8, {0}, uint8_t{0}));  // qq_bias
    pa_inputs.push_back(constant(i32, {0}, 0));                       // qq_bias_begins
    if (options.relative_bias_distances > 0) {
        params.push_back(
            make_param(f32, PartialShape{-1, head_num, options.relative_bias_distances + 1}, "relative_bias"));
        pa_inputs.push_back(params.back());
    } else {
        pa_inputs.push_back(constant(f32, {0}, 0.f));
    }
    OPENVINO_ASSERT(pa_inputs.size() == 29);

    auto paged_attn = std::make_shared<ov::op::PagedAttentionExtension>(pa_inputs);
    paged_attn->get_rt_info()["num_k_heads"] = head_num;
    paged_attn->get_rt_info()["k_head_size"] = head_size;
    paged_attn->get_rt_info()["num_v_heads"] = head_num;
    paged_attn->get_rt_info()["v_head_size"] = head_size;

    // The cache manager is attached by transformations, which functional tests disable for TEMPLATE.
    auto handle = std::make_shared<ov::reference::paged_attention_cache::CacheManagerHandle>();
    *handle = ov::reference::paged_attention_cache::make_cache_handle(type);
    ov::reference::paged_attention_cache::set_cache_manager(paged_attn.get(), *handle);
    return std::make_shared<ov::Model>(OutputVector{paged_attn}, params);
}

std::map<std::string, ov::Tensor> make_prefill_inputs(const ModelOptions& options, size_t seq_len) {
    using ov::test::utils::create_and_fill_tensor;
    using ov::test::utils::InputGenerateData;
    const auto type = options.data_type;
    const auto batch_size = options.batch_size;
    const auto head_size = static_cast<size_t>(options.head_size);
    const auto head_num = static_cast<size_t>(options.head_num);
    const size_t tokens = batch_size * seq_len;
    const size_t block = block_size(options);
    const size_t block_nums = batch_size * ((MAX_CONTEXT_LEN + block - 1) / block);

    std::map<std::string, ov::Tensor> inputs;
    inputs["q"] = create_and_fill_tensor(type, {tokens, head_num * head_size}, InputGenerateData(-1, 2, 32, 1));
    inputs["k"] = create_and_fill_tensor(type, {tokens, head_num * head_size}, InputGenerateData(-1, 2, 32, 2));
    inputs["v"] = create_and_fill_tensor(type, {tokens, head_num * head_size}, InputGenerateData(-1, 2, 32, 3));
    inputs["key_cache.0"] =
        create_and_fill_tensor(type,
                               options.cpu_layout ? ov::Shape{block_nums, head_num, block, head_size}
                                                  : ov::Shape{block_nums, head_num, head_size, block},
                               InputGenerateData(-1, 2, 32, 5));
    inputs["value_cache.0"] =
        create_and_fill_tensor(type, {block_nums, head_num, block, head_size}, InputGenerateData(-1, 2, 32, 6));
    if (options.relative_bias_distances > 0) {
        inputs["relative_bias"] =
            create_and_fill_tensor(ov::element::f32,
                                   {tokens, head_num, static_cast<size_t>(options.relative_bias_distances) + 1},
                                   InputGenerateData(-2, 4, 64, 4));
    }

    const auto blocks_per_sequence = static_cast<int32_t>((seq_len + block - 1) / block);
    const auto total_blocks = static_cast<size_t>(blocks_per_sequence) * batch_size;
    ov::Tensor past_lens(ov::element::i32, {batch_size});
    ov::Tensor subsequence_begins(ov::element::i32, {batch_size + 1});
    ov::Tensor block_indices(ov::element::i32, {total_blocks});
    ov::Tensor block_indices_begins(ov::element::i32, {batch_size + 1});
    subsequence_begins.data<int32_t>()[0] = 0;
    block_indices_begins.data<int32_t>()[0] = 0;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        past_lens.data<int32_t>()[batch] = 0;
        subsequence_begins.data<int32_t>()[batch + 1] = static_cast<int32_t>((batch + 1) * seq_len);
        block_indices_begins.data<int32_t>()[batch + 1] = static_cast<int32_t>(batch + 1) * blocks_per_sequence;
    }
    for (size_t i = 0; i < total_blocks; ++i) {
        block_indices.data<int32_t>()[i] = static_cast<int32_t>(i);
    }
    inputs["past_lens"] = past_lens;
    inputs["subsequence_begins"] = subsequence_begins;
    inputs["block_indices"] = block_indices;
    inputs["block_indices_begins"] = block_indices_begins;
    return inputs;
}

}  // namespace paged_attention_prefill
}  // namespace test
}  // namespace ov
