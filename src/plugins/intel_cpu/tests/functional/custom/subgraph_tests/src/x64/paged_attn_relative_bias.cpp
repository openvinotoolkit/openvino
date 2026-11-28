// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace {

constexpr size_t heads = 2;
constexpr size_t width = 4;
constexpr size_t span = 3;

std::shared_ptr<ov::Model> make_model(int32_t sliding_window,
                                      int32_t max_context = 64,
                                      const ov::element::Type& cache_type = ov::element::f32,
                                      float sink_value = 0.f) {
    using namespace ov;
    auto parameter = [](element::Type type, const PartialShape& shape, const std::string& name) {
        auto result = std::make_shared<op::v0::Parameter>(type, shape);
        result->set_friendly_name(name);
        result->output(0).set_names({name});
        return result;
    };
    auto q = parameter(element::f32, {-1, heads * width}, "q");
    auto k = parameter(element::f32, {-1, width}, "k");
    auto v = parameter(element::f32, {-1, width}, "v");
    auto key_cache = parameter(cache_type, {-1, 1, 32, width}, "key_cache.0");
    auto value_cache = parameter(cache_type, {-1, 1, 32, width}, "value_cache.0");
    auto past = parameter(element::i32, {-1}, "past_lens");
    auto begins = parameter(element::i32, {-1}, "subsequence_begins");
    auto blocks = parameter(element::i32, {-1}, "block_indices");
    auto block_begins = parameter(element::i32, {-1}, "block_indices_begins");
    auto bias = parameter(element::f16, {-1, heads, span + 1}, "relative_bias");
    enable_keep_const_precision(bias);

    const auto scalar_i32 = [](int32_t value) {
        return op::v0::Constant::create(element::i32, {}, {value});
    };
    const auto empty_i32 = op::v0::Constant::create(element::i32, {0}, std::vector<int32_t>{});
    const auto empty_f32 = op::v0::Constant::create(element::f32, {0}, std::vector<float>{});
    const auto empty_u8 = op::v0::Constant::create(element::u8, {0}, std::vector<uint8_t>{});
    const auto sinks = op::v0::Constant::create(element::f32, {1, heads, 1, 1}, {sink_value});
    OutputVector args{q,
                      k,
                      v,
                      key_cache,
                      value_cache,
                      past,
                      begins,
                      blocks,
                      block_begins,
                      op::v0::Constant::create(element::f32, {}, {0.5f}),
                      scalar_i32(sliding_window),
                      empty_f32,
                      scalar_i32(max_context),
                      empty_i32,
                      empty_i32,
                      empty_i32,
                      empty_f32,
                      empty_f32,
                      scalar_i32(0),
                      scalar_i32(0),
                      sinks,
                      scalar_i32(0),
                      empty_i32,
                      empty_i32,
                      empty_i32,
                      empty_i32,
                      empty_u8,
                      empty_i32,
                      bias};
    auto attention = std::make_shared<op::PagedAttentionExtension>(args);
    attention->get_rt_info()["num_k_heads"] = size_t{1};
    attention->get_rt_info()["num_v_heads"] = size_t{1};
    attention->get_rt_info()["k_head_size"] = width;
    attention->get_rt_info()["v_head_size"] = width;
    return std::make_shared<Model>(
        OutputVector{attention},
        ParameterVector{q, k, v, key_cache, value_cache, past, begins, blocks, block_begins, bias});
}

template <typename T>
ov::Tensor tensor(const ov::element::Type& type, const ov::Shape& shape, const std::vector<T>& values) {
    ov::Tensor result(type, shape);
    EXPECT_EQ(result.get_size(), values.size());
    std::copy(values.begin(), values.end(), result.data<T>());
    return result;
}

struct History {
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
};

using PackedParams = std::tuple<int32_t, ov::element::Type, ov::element::Type>;

class PagedAttnRelativeBiasCustomTest : public testing::TestWithParam<PackedParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PackedParams>& info) {
        const auto& [window, inference, cache] = info.param;
        return "window=" + std::to_string(window) + "_inference=" + inference.get_type_name() +
               "_cache=" + cache.get_type_name();
    }
};

class PagedAttnRelativeBiasLongContextTest : public testing::TestWithParam<int32_t> {};

ov::Tensor zero_cache(const ov::CompiledModel& compiled, const std::string& name, size_t blocks) {
    auto shape = compiled.input(name).get_partial_shape();
    shape[0] = static_cast<int64_t>(blocks);
    ov::Tensor cache(compiled.input(name).get_element_type(), shape.to_shape());
    std::memset(cache.data(), 0, cache.get_byte_size());
    return cache;
}

TEST_P(PagedAttnRelativeBiasCustomTest, PackedPrefillDecodeAndReuse) {
    const auto& [window, inference_precision, cache_precision] = GetParam();
    constexpr float sink = 0.7f;
    const float tolerance = inference_precision != ov::element::f32 ? 3e-2f
                            : cache_precision != ov::element::f32   ? 2e-2f
                                                                    : 2e-5f;
    auto model = make_model(window, 64, ov::element::f32, sink);
    auto compiled = ov::Core().compile_model(model,
                                             "CPU",
                                             ov::hint::inference_precision(inference_precision),
                                             ov::hint::kv_cache_precision(cache_precision));
    auto request = compiled.create_infer_request();
    request.set_tensor("key_cache.0", zero_cache(compiled, "key_cache.0", 2));
    request.set_tensor("value_cache.0", zero_cache(compiled, "value_cache.0", 2));
    request.set_tensor("block_indices", tensor<int32_t>(ov::element::i32, {2}, {0, 1}));
    request.set_tensor("block_indices_begins", tensor<int32_t>(ov::element::i32, {3}, {0, 1, 2}));

    std::vector<History> history(2);
    auto step = [&](const std::vector<int32_t>& lengths) {
        const auto count = static_cast<size_t>(lengths[0] + lengths[1]);
        std::vector<float> q(count * heads * width), k(count * width), v(count * width);
        std::vector<ov::float16> bias(count * heads * (span + 1));
        std::vector<int32_t> past(2);
        std::vector<int32_t> begins{0, lengths[0], lengths[0] + lengths[1]};
        for (size_t seq = 0; seq < 2; ++seq) {
            past[seq] = static_cast<int32_t>(history[seq].keys.size());
            for (int32_t local = 0; local < lengths[seq]; ++local) {
                const auto token = static_cast<size_t>(begins[seq] + local);
                const auto position = static_cast<size_t>(past[seq] + local);
                for (size_t d = 0; d < width; ++d) {
                    k[token * width + d] = 0.1f * static_cast<float>(1 + seq + position + d);
                    v[token * width + d] = 0.2f * static_cast<float>(1 + 2 * seq + position + d);
                    for (size_t h = 0; h < heads; ++h)
                        q[(token * heads + h) * width + d] = 0.05f * static_cast<float>(1 + seq + position + 2 * h + d);
                }
                for (size_t h = 0; h < heads; ++h)
                    for (size_t d = 0; d <= span; ++d)
                        bias[(token * heads + h) * (span + 1) + d] =
                            ov::float16(0.13f * static_cast<float>(position + h) - 0.21f * static_cast<float>(d));
            }
        }
        request.set_tensor("q", tensor<float>(ov::element::f32, {count, heads * width}, q));
        request.set_tensor("k", tensor<float>(ov::element::f32, {count, width}, k));
        request.set_tensor("v", tensor<float>(ov::element::f32, {count, width}, v));
        request.set_tensor("relative_bias", tensor<ov::float16>(ov::element::f16, {count, heads, span + 1}, bias));
        request.set_tensor("past_lens", tensor<int32_t>(ov::element::i32, {2}, past));
        request.set_tensor("subsequence_begins", tensor<int32_t>(ov::element::i32, {3}, begins));
        request.infer();
        const auto actual = request.get_output_tensor(0);
        ASSERT_EQ(actual.get_size(), count * heads * width);
        for (size_t seq = 0; seq < 2; ++seq) {
            for (int32_t local = 0; local < lengths[seq]; ++local) {
                const auto token = static_cast<size_t>(begins[seq] + local);
                history[seq].keys.emplace_back(k.begin() + token * width, k.begin() + (token + 1) * width);
                history[seq].values.emplace_back(v.begin() + token * width, v.begin() + (token + 1) * width);
                const auto n = history[seq].keys.size();
                const auto first = window > 0 && n > static_cast<size_t>(window) ? n - window : 0;
                for (size_t h = 0; h < heads; ++h) {
                    std::vector<float> weights(n);
                    float denominator = std::exp(sink);
                    for (size_t key = first; key < n; ++key) {
                        float score = 0.f;
                        for (size_t d = 0; d < width; ++d)
                            score += q[(token * heads + h) * width + d] * history[seq].keys[key][d];
                        const auto distance = std::min(n - 1 - key, span);
                        score = score * 0.5f + static_cast<float>(bias[(token * heads + h) * (span + 1) + distance]);
                        weights[key] = std::exp(score);
                        denominator += weights[key];
                    }
                    for (size_t d = 0; d < width; ++d) {
                        float expected = 0.f;
                        for (size_t key = first; key < n; ++key)
                            expected += weights[key] * history[seq].values[key][d] / denominator;
                        EXPECT_NEAR(actual.data<float>()[(token * heads + h) * width + d], expected, tolerance)
                            << "sequence=" << seq << " token=" << local << " head=" << h << " dim=" << d;
                    }
                }
            }
        }
    };
    step({3, 2});
    step({1, 1});
    step({2, 1});
    history = std::vector<History>(2);
    request.set_tensor("key_cache.0", zero_cache(compiled, "key_cache.0", 2));
    request.set_tensor("value_cache.0", zero_cache(compiled, "value_cache.0", 2));
    step({3, 2});
}

void run_long_cached_context(int32_t window, const ov::element::Type& cache_type) {
    constexpr int32_t past = 8192;
    constexpr size_t block_size = 32;
    constexpr size_t blocks_count = (past + 1 + block_size - 1) / block_size;
    auto compiled = ov::Core().compile_model(make_model(window, past + 1, cache_type),
                                             "CPU",
                                             ov::hint::inference_precision(ov::element::f32),
                                             ov::hint::kv_cache_precision(cache_type));
    auto request = compiled.create_infer_request();
    std::vector<float> keys(blocks_count * block_size * width), values(keys.size());
    for (size_t token = 0; token < past; ++token) {
        for (size_t d = 0; d < width; ++d) {
            keys[token * width + d] = 0.1f * std::sin(static_cast<float>(token) * 0.01f + static_cast<float>(d));
            values[token * width + d] = 0.3f * std::cos(static_cast<float>(token) * 0.02f + static_cast<float>(d));
        }
    }
    const std::vector<float> q{0.2f, -0.4f, 0.3f, -0.1f, -0.3f, 0.2f, 0.4f, 0.1f};
    const std::vector<float> k{0.1f, 0.3f, -0.2f, 0.4f};
    const std::vector<float> v{-0.3f, 0.2f, 0.1f, 0.5f};
    const std::vector<ov::float16> bias{0.4f, -0.2f, 0.1f, -0.3f, -0.1f, 0.3f, -0.4f, 0.2f};
    std::copy(k.begin(), k.end(), keys.begin() + past * width);
    std::copy(v.begin(), v.end(), values.begin() + past * width);
    std::vector<int32_t> blocks(blocks_count);
    std::iota(blocks.begin(), blocks.end(), 0);
    request.set_tensor("q", tensor<float>(ov::element::f32, {1, heads * width}, q));
    request.set_tensor("k", tensor<float>(ov::element::f32, {1, width}, k));
    request.set_tensor("v", tensor<float>(ov::element::f32, {1, width}, v));
    if (cache_type == ov::element::f16) {
        std::vector<ov::float16> half_keys(keys.size()), half_values(values.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            half_keys[i] = ov::float16(keys[i]);
            half_values[i] = ov::float16(values[i]);
            keys[i] = static_cast<float>(half_keys[i]);
            values[i] = static_cast<float>(half_values[i]);
        }
        request.set_tensor("key_cache.0",
                           tensor<ov::float16>(cache_type, {blocks_count, 1, block_size, width}, half_keys));
        request.set_tensor("value_cache.0",
                           tensor<ov::float16>(cache_type, {blocks_count, 1, block_size, width}, half_values));
    } else {
        request.set_tensor("key_cache.0", tensor<float>(cache_type, {blocks_count, 1, block_size, width}, keys));
        request.set_tensor("value_cache.0", tensor<float>(cache_type, {blocks_count, 1, block_size, width}, values));
    }
    request.set_tensor("past_lens", tensor<int32_t>(ov::element::i32, {1}, {past}));
    request.set_tensor("subsequence_begins", tensor<int32_t>(ov::element::i32, {2}, {0, 1}));
    request.set_tensor("block_indices", tensor<int32_t>(ov::element::i32, {blocks_count}, blocks));
    request.set_tensor("block_indices_begins",
                       tensor<int32_t>(ov::element::i32, {2}, {0, static_cast<int32_t>(blocks_count)}));
    request.set_tensor("relative_bias", tensor<ov::float16>(ov::element::f16, {1, heads, span + 1}, bias));
    ASSERT_NO_THROW(request.infer());

    const auto actual = request.get_output_tensor(0);
    ASSERT_EQ(actual.get_size(), heads * width);
    const size_t first = window ? past + 1 - window : 0;
    for (size_t h = 0; h < heads; ++h) {
        std::vector<float> weights(past + 1);
        float max_score = 0.f;
        for (size_t token = first; token <= past; ++token) {
            float score = 0.f;
            for (size_t d = 0; d < width; ++d)
                score += q[h * width + d] * keys[token * width + d];
            score = score * 0.5f + static_cast<float>(bias[h * (span + 1) + std::min(past - token, span)]);
            weights[token] = score;
            max_score = std::max(max_score, score);
        }
        float denominator = std::exp(-max_score);
        for (size_t token = first; token <= past; ++token) {
            weights[token] = std::exp(weights[token] - max_score);
            denominator += weights[token];
        }
        for (size_t d = 0; d < width; ++d) {
            float expected = 0.f;
            for (size_t token = first; token <= past; ++token)
                expected += weights[token] * values[token * width + d] / denominator;
            EXPECT_NEAR(actual.data<const float>()[h * width + d], expected, 2e-5f) << "head=" << h << " dim=" << d;
        }
    }
}

TEST_P(PagedAttnRelativeBiasLongContextTest, LongCachedContext) {
    for (const auto& cache_type : {ov::element::f32, ov::element::f16}) {
        SCOPED_TRACE(cache_type);
        run_long_cached_context(GetParam() ? 1024 : 0, cache_type);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnRelativeBias,
                         PagedAttnRelativeBiasCustomTest,
                         testing::Combine(testing::Values(0, 2),
                                          testing::Values(ov::element::f32, ov::element::bf16, ov::element::f16),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::u8)),
                         PagedAttnRelativeBiasCustomTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnRelativeBiasLongContext,
                         PagedAttnRelativeBiasLongContextTest,
                         testing::Values(0, 2));

}  // namespace
