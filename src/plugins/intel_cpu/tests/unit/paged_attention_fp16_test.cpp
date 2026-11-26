// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

namespace ov::intel_cpu {
namespace {

struct Params {
    std::vector<size_t> lengths;
    size_t heads;
    size_t kv_heads;
    size_t key_size;
    size_t value_size;
    ov::element::Type cache_type;
};
class PagedAttentionFP16Test : public testing::TestWithParam<std::tuple<Params, bool>> {};

TEST(ScaledDotProductAttentionFP16Test, PrefillWithKeyAndValueTailsMatchesReference) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16)) {
        GTEST_SKIP() << "FP16 attention requires AVX512-FP16.";
    }
    constexpr size_t batch = 2;
    constexpr size_t heads = 4;
    constexpr size_t query_len = 33;
    constexpr size_t kv_len = 65;
    constexpr size_t key_size = 65;
    constexpr size_t value_size = 35;
    std::vector<ov::Tensor> inputs{ov::Tensor(ov::element::f16, {batch, heads, query_len, key_size}),
                                   ov::Tensor(ov::element::f16, {batch, heads, kv_len, key_size}),
                                   ov::Tensor(ov::element::f16, {batch, heads, kv_len, value_size})};
    ov::ParameterVector parameters;
    for (size_t i = 0; i < inputs.size(); ++i) {
        ov::PartialShape shape(inputs[i].get_shape());
        shape[2] = ov::Dimension::dynamic();
        parameters.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        for (size_t j = 0; j < inputs[i].get_size(); ++j) {
            inputs[i].data<ov::float16>()[j] =
                ov::float16(static_cast<float>(static_cast<int>((j * 7 + i * 3) % 17) - 8) / 32.0F);
        }
    }
    auto attention =
        std::make_shared<ov::op::v13::ScaledDotProductAttention>(parameters[0], parameters[1], parameters[2], false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{attention}, parameters);
    ov::Core core;
    auto compiled = core.compile_model(
        model,
        "CPU",
        {ov::hint::inference_precision(ov::element::f16), ov::num_streams(1), ov::inference_num_threads(2)});
    auto request = compiled.create_infer_request();
    for (size_t i = 0; i < inputs.size(); ++i) {
        request.set_input_tensor(i, inputs[i]);
    }
    request.infer();
    auto output = request.get_output_tensor();
    ASSERT_EQ(output.get_element_type(), ov::element::f16);
    ASSERT_EQ(output.get_shape(), (ov::Shape{batch, heads, query_len, value_size}));
    const float scale = 1.0F / std::sqrt(static_cast<float>(key_size));
    for (size_t bh = 0; bh < batch * heads; ++bh) {
        for (size_t m = 0; m < query_len; ++m) {
            std::vector<float> weights(kv_len, 0.0F);
            for (size_t n = 0; n < kv_len; ++n) {
                for (size_t k = 0; k < key_size; ++k) {
                    weights[n] +=
                        static_cast<float>(inputs[0].data<ov::float16>()[(bh * query_len + m) * key_size + k]) *
                        static_cast<float>(inputs[1].data<ov::float16>()[(bh * kv_len + n) * key_size + k]);
                }
                weights[n] *= scale;
            }
            const float maximum = *std::max_element(weights.begin(), weights.end());
            float denominator = 0.0F;
            for (auto& weight : weights) {
                weight = std::exp(weight - maximum);
                denominator += weight;
            }
            for (auto& weight : weights) {
                weight = static_cast<float>(ov::float16(weight / denominator));
            }
            for (size_t c = 0; c < value_size; ++c) {
                float expected = 0.0F;
                for (size_t n = 0; n < kv_len; ++n) {
                    expected += weights[n] *
                                static_cast<float>(inputs[2].data<ov::float16>()[(bh * kv_len + n) * value_size + c]);
                }
                ASSERT_NEAR(static_cast<float>(output.data<ov::float16>()[(bh * query_len + m) * value_size + c]),
                            expected,
                            0.002F);
            }
        }
    }
}

TEST_P(PagedAttentionFP16Test, PrefillAndDecodeMatchReference) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16)) {
        GTEST_SKIP() << "FP16 PagedAttention requires AVX512-FP16.";
    }
    const auto& dimensions = std::get<0>(GetParam());
    const bool nonzero_qk = std::get<1>(GetParam());
    const auto& [lengths, heads, kv_heads, key_size, value_size, cache_type] = dimensions;
    const size_t sequences = lengths.size();
    const size_t tokens = std::accumulate(lengths.begin(), lengths.end(), size_t{0});
    std::vector<size_t> continuation(sequences, 1);
    if (sequences > 1) {
        continuation[1] = 3;
    }
    constexpr size_t block_size = 32;
    std::vector<size_t> block_begins(sequences + 1, 0);
    size_t max_context = 0;
    for (size_t sequence = 0; sequence < sequences; ++sequence) {
        const size_t capacity = lengths[sequence] + continuation[sequence];
        block_begins[sequence + 1] = block_begins[sequence] + (capacity + block_size - 1) / block_size;
        max_context = std::max(max_context, capacity);
    }
    const size_t blocks = block_begins.back();
    auto tensor = [](ov::element::Type type, const ov::Shape& shape) {
        ov::Tensor result(type, shape);
        std::fill_n(static_cast<uint8_t*>(result.data()), result.get_byte_size(), uint8_t{0});
        return result;
    };
    auto scalar = [&](int32_t value) {
        auto result = tensor(ov::element::i32, {});
        result.data<int32_t>()[0] = value;
        return result;
    };
    std::vector<ov::Tensor> inputs(28);
    for (auto& input : inputs) {
        input = tensor(ov::element::i32, {0});
    }
    inputs[0] = tensor(ov::element::f16, {tokens, heads * key_size});
    inputs[1] = tensor(ov::element::f16, {tokens, kv_heads * key_size});
    inputs[2] = tensor(ov::element::f16, {tokens, kv_heads * value_size});
    inputs[3] = tensor(ov::element::f16, {blocks, kv_heads, block_size, key_size});
    inputs[4] = tensor(ov::element::f16, {blocks, kv_heads, block_size, value_size});
    inputs[5] = tensor(ov::element::i32, {sequences});
    inputs[6] = tensor(ov::element::i32, {sequences + 1});
    inputs[7] = tensor(ov::element::i32, {blocks});
    for (size_t i = 0; i < blocks; ++i) {
        inputs[7].data<int32_t>()[i] = static_cast<int32_t>(blocks - 1 - i);
    }
    inputs[8] = tensor(ov::element::i32, {sequences + 1});
    for (size_t sequence = 0; sequence <= sequences; ++sequence) {
        inputs[8].data<int32_t>()[sequence] = static_cast<int32_t>(block_begins[sequence]);
    }
    inputs[9] = tensor(ov::element::f32, {});
    inputs[9].data<float>()[0] = 1.0F;
    inputs[10] = scalar(0);
    inputs[11] = tensor(ov::element::f32, {0});
    inputs[12] = scalar(static_cast<int32_t>(max_context));
    inputs[13] = tensor(ov::element::i32, {sequences});
    inputs[16] = tensor(ov::element::f32, {0});
    inputs[17] = tensor(ov::element::f32, {0});
    inputs[18] = scalar(64);
    inputs[19] = scalar(8);
    inputs[20] = tensor(ov::element::f16, {0});
    inputs[21] = scalar(0);
    inputs[26] = tensor(ov::element::u8, {0});
    ov::OutputVector arguments;
    ov::ParameterVector parameters;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i < 9) {
            ov::PartialShape shape(inputs[i].get_shape());
            if (i < 3) {
                shape[0] = ov::Dimension::dynamic();
            }
            auto parameter = std::make_shared<ov::op::v0::Parameter>(inputs[i].get_element_type(), shape);
            parameters.push_back(parameter);
            arguments.push_back(parameter);
        } else {
            arguments.push_back(std::make_shared<ov::op::v0::Constant>(inputs[i]));
        }
    }
    auto attention = std::make_shared<ov::op::PagedAttentionExtension>(arguments);
    attention->get_rt_info()["k_head_size"] = static_cast<int64_t>(key_size);
    attention->get_rt_info()["num_k_heads"] = static_cast<int64_t>(kv_heads);
    attention->get_rt_info()["v_head_size"] = static_cast<int64_t>(value_size);
    attention->get_rt_info()["num_v_heads"] = static_cast<int64_t>(kv_heads);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{attention->output(0)}, parameters);
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       {ov::hint::inference_precision(ov::element::f16),
                                        ov::hint::kv_cache_precision(cache_type),
                                        ov::intel_cpu::enable_sage_attn(false),
                                        ov::num_streams(1),
                                        ov::inference_num_threads(2)});
    auto request = compiled.create_infer_request();
    for (size_t i : {size_t{3}, size_t{4}}) {
        auto shape = compiled.input(i).get_partial_shape();
        shape[0] = blocks;
        inputs[i] = tensor(compiled.input(i).get_element_type(), shape.to_shape());
    }
    auto value = [cache_type](size_t sequence, size_t token, size_t channel) {
        const size_t levels = cache_type == ov::element::f16 ? 17 : 16;
        return static_cast<float>(static_cast<int>((sequence * 5 + token * 7 + channel * 3) % levels) - 8) / 16.0F;
    };
    auto query = [nonzero_qk](size_t sequence, size_t token, size_t channel) {
        return nonzero_qk
                   ? static_cast<float>(static_cast<int>((sequence * 3 + token * 3 + channel * 7) % 11) - 5) / 32.0F
                   : 0.0F;
    };
    auto key = [nonzero_qk](size_t sequence, size_t token, size_t channel) {
        return nonzero_qk
                   ? static_cast<float>(static_cast<int>((sequence * 7 + token * 5 + channel * 3) % 16) - 8) / 32.0F
                   : 0.0F;
    };
    std::vector<size_t> past_lengths(sequences, 0);
    auto infer = [&](const std::vector<size_t>& query_lengths) {
        const size_t query_tokens = std::accumulate(query_lengths.begin(), query_lengths.end(), size_t{0});
        inputs[0] = tensor(ov::element::f16, {query_tokens, heads * key_size});
        inputs[1] = tensor(ov::element::f16, {query_tokens, kv_heads * key_size});
        inputs[2] = tensor(ov::element::f16, {query_tokens, kv_heads * value_size});
        size_t row = 0;
        for (size_t sequence = 0; sequence < sequences; ++sequence) {
            inputs[5].data<int32_t>()[sequence] = static_cast<int32_t>(past_lengths[sequence]);
            inputs[6].data<int32_t>()[sequence] = static_cast<int32_t>(row);
            for (size_t t = 0; t < query_lengths[sequence]; ++t, ++row) {
                const size_t token = past_lengths[sequence] + t;
                for (size_t c = 0; c < heads * key_size; ++c) {
                    inputs[0].data<ov::float16>()[row * heads * key_size + c] = ov::float16(query(sequence, token, c));
                }
                for (size_t c = 0; c < kv_heads * key_size; ++c) {
                    inputs[1].data<ov::float16>()[row * kv_heads * key_size + c] = ov::float16(key(sequence, token, c));
                }
                for (size_t c = 0; c < kv_heads * value_size; ++c) {
                    inputs[2].data<ov::float16>()[row * kv_heads * value_size + c] =
                        ov::float16(value(sequence, token, c));
                }
            }
        }
        inputs[6].data<int32_t>()[sequences] = static_cast<int32_t>(row);
        for (size_t i = 0; i < parameters.size(); ++i) {
            request.set_input_tensor(i, inputs[i]);
        }
        request.infer();
    };
    auto compare = [&](size_t sequence, size_t row, size_t last_token) {
        auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_element_type(), ov::element::f16);
        for (size_t h = 0; h < heads; ++h) {
            const size_t kv_head = h / (heads / kv_heads);
            std::vector<float> weights(last_token + 1, 0.0F);
            for (size_t t = 0; t <= last_token; ++t) {
                for (size_t c = 0; c < key_size; ++c) {
                    weights[t] +=
                        query(sequence, last_token, h * key_size + c) * key(sequence, t, kv_head * key_size + c);
                }
            }
            const float max_score = *std::max_element(weights.begin(), weights.end());
            float denominator = 0.0F;
            for (auto& weight : weights) {
                weight = std::exp(weight - max_score);
                denominator += weight;
            }
            for (auto& weight : weights) {
                weight = static_cast<float>(ov::float16(weight / denominator));
            }
            for (size_t c = 0; c < value_size; ++c) {
                float expected = 0.0F;
                for (size_t t = 0; t <= last_token; ++t) {
                    expected += weights[t] * value(sequence, t, kv_head * value_size + c);
                }
                ASSERT_NEAR(
                    static_cast<float>(output.data<ov::float16>()[row * heads * value_size + h * value_size + c]),
                    expected,
                    0.002F)
                    << "sequence=" << sequence << " row=" << row << " head=" << h << " channel=" << c;
            }
        }
    };
    for (const auto& query_lengths : {lengths, continuation}) {
        infer(query_lengths);
        ASSERT_EQ(
            request.get_output_tensor().get_shape(),
            (ov::Shape{std::accumulate(query_lengths.begin(), query_lengths.end(), size_t{0}), heads * value_size}));
        size_t row = 0;
        for (size_t sequence = 0; sequence < sequences; ++sequence) {
            for (size_t t = 0; t < query_lengths[sequence]; ++t, ++row) {
                ASSERT_NO_FATAL_FAILURE(compare(sequence, row, past_lengths[sequence] + t));
            }
            past_lengths[sequence] += query_lengths[sequence];
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         PagedAttentionFP16Test,
                         testing::Combine(testing::Values(Params{{33}, 1, 1, 64, 64, ov::element::f16},
                                                          Params{{63}, 8, 8, 64, 80, ov::element::f16},
                                                          Params{{63}, 2, 2, 128, 64, ov::element::f16},
                                                          Params{{33, 63}, 8, 2, 128, 64, ov::element::f16},
                                                          Params{{33}, 1, 1, 64, 64, ov::element::u8},
                                                          Params{{63}, 8, 8, 64, 80, ov::element::u8},
                                                          Params{{63}, 2, 2, 128, 64, ov::element::u8},
                                                          Params{{33, 63}, 8, 2, 128, 64, ov::element::u8},
                                                          Params{{33}, 1, 1, 64, 64, ov::element::u4},
                                                          Params{{63}, 8, 8, 64, 80, ov::element::u4},
                                                          Params{{63}, 2, 2, 128, 64, ov::element::u4},
                                                          Params{{33, 63}, 8, 2, 128, 64, ov::element::u4}),
                                          testing::Values(false, true)));

}  // namespace
}  // namespace ov::intel_cpu
