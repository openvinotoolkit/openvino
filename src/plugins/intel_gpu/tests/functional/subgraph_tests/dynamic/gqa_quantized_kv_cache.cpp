// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

// GroupQueryAttention with a quantized KV cache (kv_cache_bit_width 8 / 4, per-channel k/v scales) executed end to end
// against the TEMPLATE plugin. On the sdpa_ocl lane the GPU GroupQueryAttentionDecomposition hands SDPA the quantized
// cache and its per-channel scales (the 4-bit cache with a fixed zero point 8), so the kernel dequantizes per channel;
// elsewhere the K/V are dequantized before SDPA. Fused are the batch-1 caches the GPU decomposition builds: the full
// StatelessKV cache and the windowed cache (sliding_window_cache, with the native SDPA window). Batch 2 takes the common
// per-batch cache, which dequantizes before SDPA on every device (the unfused control).
// The oracle is the same model in f32 on TEMPLATE (TEMPLATE runs an f16 model in f16, which was off by up to 0.06 on these
// values on every GPU path). The current K/V are integer codes times power-of-two scales, so quantize-on-write is exact
// in either precision and the present caches must match bit for bit.
namespace {

struct GQAQuantParams {
    int64_t bits;
    size_t head_size;
    size_t batch;
    bool windowed;  // sliding_window_cache with a window of `window` keys in `capacity` rows
};

constexpr size_t num_heads = 8;
constexpr size_t kv_num_heads = 2;
constexpr int64_t window = 16;
constexpr size_t capacity = 24;

size_t cache_head_size(const GQAQuantParams& p) {
    return p.bits == 4 ? p.head_size / 2 : p.head_size;
}

// Per-channel scales 2^e, e in [e_min, e_min + 3], so neighbouring channels differ by up to 8x.
std::vector<float> make_scales(const GQAQuantParams& p, std::mt19937& gen) {
    std::uniform_int_distribution<int> e(0, 3);
    const int e_min = p.bits == 8 ? -7 : -3;
    std::vector<float> scales(kv_num_heads * p.head_size);
    for (auto& s : scales) {
        s = std::ldexp(1.0f, e_min + e(gen));
    }
    return scales;
}

// `precision`: Q / K / V / output type, f16 for the GPU, f32 for the oracle.
std::shared_ptr<ov::Model> make_gqa_model(const GQAQuantParams& p,
                                          const std::vector<float>& k_scale,
                                          const std::vector<float>& v_scale,
                                          ov::element::Type precision) {
    const auto f16 = precision;
    const auto cache_type = p.bits == 8 ? ov::element::i8 : ov::element::u8;
    const auto B = static_cast<int64_t>(p.batch);
    const auto D = static_cast<int64_t>(p.head_size);
    const auto DC = static_cast<int64_t>(cache_head_size(p));
    const auto H = static_cast<int64_t>(num_heads);
    const auto KVH = static_cast<int64_t>(kv_num_heads);

    auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{B, H, -1, D});
    auto key = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{B, KVH, -1, D});
    auto value = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{B, KVH, -1, D});
    const auto cache_len = p.windowed ? ov::Dimension(static_cast<int64_t>(capacity)) : ov::Dimension::dynamic();
    auto past_key = std::make_shared<ov::op::v0::Parameter>(cache_type, ov::PartialShape{B, KVH, cache_len, DC});
    auto past_value = std::make_shared<ov::op::v0::Parameter>(cache_type, ov::PartialShape{B, KVH, cache_len, DC});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{B});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});

    ov::OutputVector inputs(14);
    inputs[0] = query;
    inputs[1] = key;
    inputs[2] = value;
    inputs[3] = past_key;
    inputs[4] = past_value;
    inputs[5] = seqlens_k;
    inputs[6] = total_sequence_length;
    for (size_t i = 7; i <= 11; ++i) {
        inputs[i] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    }
    inputs[12] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{k_scale.size()}, k_scale);
    inputs[13] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{v_scale.size()}, v_scale);
    const auto per_channel = ov::op::internal::GroupQueryAttentionQuantType::PER_CHANNEL;
    auto gqa = std::make_shared<
        ov::op::internal::GroupQueryAttention>(inputs, H, KVH, 0.0f, false, false, p.bits, per_channel, per_channel, p.windowed ? window : -1, p.windowed);
    ov::ResultVector results;
    for (size_t i = 0; i < gqa->get_output_size(); ++i) {
        results.push_back(std::make_shared<ov::op::v0::Result>(gqa->output(i)));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length});
}

ov::Tensor random_query(const ov::Shape& shape, std::mt19937& gen) {
    ov::Tensor tensor(ov::element::f16, shape);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = ov::float16(dist(gen));
    }
    return tensor;
}

// [B, KVH, S, D] current K/V whose channel c of KV head h is an integer code times scales[h * D + c].
ov::Tensor coded_kv(const GQAQuantParams& p, size_t seq, const std::vector<float>& scales, std::mt19937& gen) {
    ov::Tensor tensor(ov::element::f16, ov::Shape{p.batch, kv_num_heads, seq, p.head_size});
    std::uniform_int_distribution<int> code(p.bits == 8 ? -127 : -8, p.bits == 8 ? 127 : 7);
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        const size_t channel = i % p.head_size;
        const size_t kv_head = (i / (p.head_size * seq)) % kv_num_heads;
        data[i] = ov::float16(static_cast<float>(code(gen)) * scales[kv_head * p.head_size + channel]);
    }
    return tensor;
}

ov::Tensor random_cache(const GQAQuantParams& p, size_t past, std::mt19937& gen) {
    ov::Tensor tensor(p.bits == 8 ? ov::element::i8 : ov::element::u8, ov::Shape{p.batch, kv_num_heads, past, cache_head_size(p)});
    std::uniform_int_distribution<int> byte(p.bits == 8 ? -127 : 0, p.bits == 8 ? 127 : 255);
    auto* data = static_cast<uint8_t*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = static_cast<uint8_t>(byte(gen));
    }
    return tensor;
}

class GQAQuantizedKVCacheTest : public ::testing::TestWithParam<GQAQuantParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<GQAQuantParams>& info) {
        const auto& p = info.param;
        return "int" + std::to_string(p.bits) + "_D" + std::to_string(p.head_size) + "_B" + std::to_string(p.batch) + (p.windowed ? "_windowed" : "");
    }
};

TEST_P(GQAQuantizedKVCacheTest, MatchesTemplate) {
    const auto& p = GetParam();
    std::mt19937 gen(static_cast<uint32_t>(p.bits * 1000 + p.head_size * 10 + p.batch + (p.windowed ? 5 : 0)));
    const auto k_scale = make_scales(p, gen);
    const auto v_scale = make_scales(p, gen);
    auto core = ov::test::utils::PluginCache::get().core();
    const auto model = make_gqa_model(p, k_scale, v_scale, ov::element::f16);
    auto gpu_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto ref_model = core->compile_model(make_gqa_model(p, k_scale, v_scale, ov::element::f32), ov::test::utils::DEVICE_TEMPLATE);
    auto gpu = gpu_model.create_infer_request();
    auto ref = ref_model.create_infer_request();

    // {past, current}: a prefill, two decodes and a chunk on a past (windowed: past the window and the capacity, as in
    // smoke_GQASlidingWindow).
    const std::vector<std::pair<size_t, size_t>> steps = p.windowed ? std::vector<std::pair<size_t, size_t>>{{0, 40}, {40, 1}, {41, 1}, {42, 20}}
                                                                    : std::vector<std::pair<size_t, size_t>>{{0, 40}, {40, 1}, {300, 1}, {64, 33}};
    for (const auto& [past, current] : steps) {
        SCOPED_TRACE("past " + std::to_string(past) + ", current " + std::to_string(current));
        // The windowed cache is always `capacity` rows; the decomposition derives the resident rows from seqlens_k.
        const size_t past_rows = p.windowed ? capacity : past;
        std::vector<ov::Tensor> inputs = {
            random_query({p.batch, num_heads, current, p.head_size}, gen),
            coded_kv(p, current, k_scale, gen),
            coded_kv(p, current, v_scale, gen),
            random_cache(p, past_rows, gen),
            random_cache(p, past_rows, gen),
            ov::Tensor(ov::element::i32, ov::Shape{p.batch}),
            ov::Tensor(ov::element::i32, ov::Shape{}),
        };
        for (size_t b = 0; b < p.batch; ++b) {
            inputs[5].data<int32_t>()[b] = static_cast<int32_t>(past + current - 1);
        }
        inputs[6].data<int32_t>()[0] = static_cast<int32_t>(past + current);
        for (size_t i = 0; i < inputs.size(); ++i) {
            gpu.set_input_tensor(i, inputs[i]);
            if (inputs[i].get_element_type() == ov::element::f16) {
                ov::Tensor f32(ov::element::f32, inputs[i].get_shape());
                const auto* src = inputs[i].data<const ov::float16>();
                auto* dst = f32.data<float>();
                for (size_t j = 0; j < f32.get_size(); ++j) {
                    dst[j] = static_cast<float>(src[j]);
                }
                ref.set_input_tensor(i, f32);
            } else {
                ref.set_input_tensor(i, inputs[i]);
            }
        }
        gpu.infer();
        ref.infer();

        for (size_t o = 0; o < model->get_output_size(); ++o) {
            const auto expected = ref.get_output_tensor(o);
            const auto actual = gpu.get_output_tensor(o);
            ASSERT_EQ(expected.get_shape(), actual.get_shape()) << "output " << o;
            if (o > 0) {
                ASSERT_EQ(expected.get_element_type(), actual.get_element_type()) << "output " << o;
                // Present caches: copied past bytes plus exactly quantized new rows.
                const auto* e = static_cast<const uint8_t*>(expected.data());
                const auto* a = static_cast<const uint8_t*>(actual.data());
                size_t mismatches = 0;
                for (size_t i = 0; i < expected.get_byte_size(); ++i) {
                    mismatches += e[i] != a[i];
                }
                EXPECT_EQ(mismatches, 0u) << "output " << o << ": " << mismatches << " of " << expected.get_byte_size() << " bytes differ";
                continue;
            }
            // Attention output: f16 compute on values up to 8, against f32.
            const auto* e = expected.data<const float>();
            const auto* a = actual.data<const ov::float16>();
            size_t mismatches = 0;
            float max_diff = 0.0f;
            size_t first_mismatch = 0;
            for (size_t i = 0; i < expected.get_size(); ++i) {
                const float ev = static_cast<float>(e[i]);
                const float diff = std::fabs(ev - static_cast<float>(a[i]));
                if (!(diff <= 0.02f + 0.01f * std::fabs(ev))) {  // NaN counts as a mismatch
                    if (mismatches++ == 0) {
                        first_mismatch = i;
                    }
                }
                if (diff > max_diff || std::isnan(diff)) {
                    max_diff = diff;
                }
            }
            EXPECT_EQ(mismatches, 0u) << "output 0: " << mismatches << " of " << expected.get_size() << " elements off, max diff " << max_diff << ", first at "
                                      << first_mismatch << " (expected " << static_cast<float>(e[first_mismatch]) << ", actual "
                                      << static_cast<float>(a[first_mismatch]) << ")";
        }
    }
}

std::vector<GQAQuantParams> make_params() {
    std::vector<GQAQuantParams> params;
    for (const int64_t bits : {8, 4}) {
        // 80: the i8 rows (80 B) and the 4-bit ones (40 B) miss the block-read pitch rule, so both take the gathers.
        for (const size_t head_size : {64, 80, 128}) {
            for (const size_t batch : {1, 2}) {
                params.push_back({bits, head_size, batch, false});
            }
            params.push_back({bits, head_size, 1, true});
        }
        // Larger heads: 256 is the largest the decomposition fuses, 512 stays unfused (dequantized before SDPA).
        for (const size_t head_size : {256, 512}) {
            params.push_back({bits, head_size, 1, false});
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_GQAQuantizedKVCache, GQAQuantizedKVCacheTest, ::testing::ValuesIn(make_params()), GQAQuantizedKVCacheTest::get_test_case_name);

}  // namespace
