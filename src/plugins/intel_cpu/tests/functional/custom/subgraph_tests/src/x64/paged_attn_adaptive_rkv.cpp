// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <cfloat>
#include <numeric>

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "internal_properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/reference/adaptive_rkv_diversity.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {

struct DiversityTestParams {
    size_t head_size;
    size_t head_num;
    size_t block_size;
    size_t start_size;
    size_t eviction_size;
    size_t num_data_blocks;
    ov::element::Type kv_cache_precision;
    bool force_by_token;
    std::string name;
};

class PagedAttnAdaptiveRKVDiversityTest : public testing::WithParamInterface<DiversityTestParams>,
                                          virtual public SubgraphBaseTest,
                                          public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DiversityTestParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << p.name;
        if (p.kv_cache_precision != ov::element::f32) {
            result << "_" << p.kv_cache_precision.get_type_name();
            if (p.force_by_token) {
                result << "_bytoken";
            }
        }
        return result.str();
    }

    // model parameters
    size_t kHeadSize;
    size_t kHeadNum;
    // paged attention parameters
    size_t kBlockSize;
    size_t kNumDataBlocks;
    size_t kNumBlocks;
    size_t kTokenCount;
    // adaptive RKV parameters
    size_t kStartSize;
    size_t kEvictionSize;
    // cache quantization parameters
    ov::element::Type kKVCachePrecision;
    bool kForceByToken;
    bool kQuantByChannel;

    static std::shared_ptr<ov::op::v0::Parameter> make_param(const PartialShape& pshape,
                                                             element::Type element_type,
                                                             const std::string& name) {
        auto param = std::make_shared<v0::Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }

    void SetUp() override {
        const auto& p = GetParam();
        kHeadSize = p.head_size;
        kHeadNum = p.head_num;
        kBlockSize = p.block_size;
        kStartSize = p.start_size;
        kEvictionSize = p.eviction_size;
        kNumDataBlocks = p.num_data_blocks;
        kNumBlocks = kNumDataBlocks + 1;
        kTokenCount = kNumDataBlocks * kBlockSize;
        kKVCachePrecision = p.kv_cache_precision;
        kForceByToken = p.force_by_token;
        kQuantByChannel = kKVCachePrecision.is_integral() && !kForceByToken && kKVCachePrecision != ov::element::i8;

        targetDevice = utils::DEVICE_CPU;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        if (kKVCachePrecision == ov::element::i8) {
            // i8 key cache requires SageAttn; it forces key=i8 by-token, value=u8 by-token
            configuration[ov::intel_cpu::enable_sage_attn.name()] = true;
        } else if (kKVCachePrecision == ov::element::u4) {
            configuration[ov::key_cache_precision.name()] = ov::element::u4;
            configuration[ov::value_cache_precision.name()] = ov::element::u4;
        } else {
            configuration[ov::hint::kv_cache_precision.name()] = kKVCachePrecision;
        }
        if (kKVCachePrecision.is_integral() && kForceByToken) {
            configuration[ov::internal::key_cache_quant_mode.name()] = ov::internal::CacheQuantMode::BY_TOKEN;
            configuration[ov::internal::value_cache_quant_mode.name()] = ov::internal::CacheQuantMode::BY_TOKEN;
        }
        function = build_model();
    }

    std::shared_ptr<ov::Model> build_model() {
        const auto hs = static_cast<int64_t>(kHeadSize);
        const auto hn = static_cast<int64_t>(kHeadNum);
        const auto bs = static_cast<int64_t>(kBlockSize);
        const auto nb = static_cast<int64_t>(kNumBlocks);

        auto q = make_param(PartialShape{1, hs * hn}, ov::element::f32, "q");
        auto k = make_param(PartialShape{1, hs * hn}, ov::element::f32, "k");
        auto v = make_param(PartialShape{1, hs * hn}, ov::element::f32, "v");
        auto key_cache = make_param(PartialShape{nb, hn, bs, hs}, ov::element::f32, "key_cache.0");
        auto value_cache = make_param(PartialShape{nb, hn, bs, hs}, ov::element::f32, "value_cache.0");
        auto past_lens = make_param(PartialShape{1}, ov::element::i32, "past_lens");
        auto subsequence_begins = make_param(PartialShape{2}, ov::element::i32, "subsequence_begins");
        auto block_indices = make_param(PartialShape{nb}, ov::element::i32, "block_indices");
        auto block_indices_begins = make_param(PartialShape{2}, ov::element::i32, "block_indices_begins");

        auto scale = std::make_shared<v0::Constant>(ov::element::f32, Shape{}, std::vector<float>{0.5f});
        auto sliding_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto alibi_slopes = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto max_context_len = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{static_cast<int32_t>(kNumBlocks * kBlockSize)});
        auto score_aggregation_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{});
        auto rotated_block_indices = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{});
        auto rotation_deltas = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{});
        auto rotation_trig_lut = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto xattention_threshold = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto xattention_block_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{64});
        auto xattention_stride = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{8});
        auto sinks = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto adaptive_rkv_start_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{static_cast<int32_t>(kStartSize)});
        auto adaptive_rkv_evictable_sizes = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, std::vector<int32_t>{static_cast<int32_t>(kEvictionSize)});
        std::vector<int32_t> block_set_indices_data(kNumDataBlocks);
        std::iota(block_set_indices_data.begin(), block_set_indices_data.end(), 0);
        auto adaptive_rkv_diversity_block_set_indices =
            std::make_shared<v0::Constant>(ov::element::i32, Shape{kNumDataBlocks}, block_set_indices_data);
        auto adaptive_rkv_diversity_block_set_indices_begins =
            std::make_shared<v0::Constant>(ov::element::i32, Shape{2}, std::vector<int32_t>{0, static_cast<int32_t>(kNumDataBlocks)});
        auto token_type_ids = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{});
        auto qq_bias = std::make_shared<v0::Constant>(ov::element::u8, Shape{0}, std::vector<uint8_t>{0});
        auto qq_bias_begins = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});

        OutputVector inputs = {q,
                               k,
                               v,
                               key_cache,
                               value_cache,
                               past_lens,
                               subsequence_begins,
                               block_indices,
                               block_indices_begins,
                               scale,
                               sliding_window,
                               alibi_slopes,
                               max_context_len,
                               score_aggregation_window,
                               rotated_block_indices,
                               rotation_deltas,
                               rotation_trig_lut,
                               xattention_threshold,
                               xattention_block_size,
                               xattention_stride,
                               sinks,
                               adaptive_rkv_start_size,
                               adaptive_rkv_evictable_sizes,
                               adaptive_rkv_diversity_block_set_indices,
                               adaptive_rkv_diversity_block_set_indices_begins,
                               token_type_ids,
                               qq_bias,
                               qq_bias_begins};

        auto paged_attn = std::make_shared<op::PagedAttentionExtension>(inputs);
        paged_attn->get_rt_info()["num_k_heads"] = kHeadNum;
        paged_attn->get_rt_info()["k_head_size"] = kHeadSize;
        paged_attn->get_rt_info()["num_v_heads"] = kHeadNum;
        paged_attn->get_rt_info()["v_head_size"] = kHeadSize;

        return std::make_shared<ov::Model>(OutputVector{paged_attn->output(0), paged_attn->output(1), paged_attn->output(2)},
                                           ParameterVector{q,
                                                           k,
                                                           v,
                                                           key_cache,
                                                           value_cache,
                                                           past_lens,
                                                           subsequence_begins,
                                                           block_indices,
                                                           block_indices_begins});
    }

    static float make_key_value(size_t token_idx, size_t dim) {
        const float sign = (token_idx + dim) % 2 == 0 ? 1.0f : -1.0f;
        return 0.25f * static_cast<float>(dim + 1) + 0.03125f * static_cast<float>(token_idx + 1) +
               sign * 0.0625f * static_cast<float>((token_idx % 7) + 1);
    }

    // ── Scalar quantize/dequantize helpers ──────────────────────────────

    static void scalar_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
        float max_val = -FLT_MAX, min_val = FLT_MAX;
        for (size_t i = 0; i < n; i++) {
            max_val = std::max(max_val, src[i]);
            min_val = std::min(min_val, src[i]);
        }
        scale = (max_val - min_val) / 255.0f;
        if (scale == 0.0f) scale = 0.0001f;
        zp = -min_val / scale;
        for (size_t i = 0; i < n; i++) {
            dst[i] = static_cast<uint8_t>(std::round(std::max(src[i] / scale + zp, 0.0f)));
        }
    }

    static void scalar_dequant_u8(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = (static_cast<float>(src[i]) - zp) * scale;
        }
    }

    static void scalar_quant_u8_by_channel(const float* src, uint8_t* dst,
                                           size_t seq_dim, size_t hidden_dims,
                                           size_t dst_stride,
                                           float* scale, float* zp) {
        for (size_t j = 0; j < hidden_dims; j++) {
            float max_val = -FLT_MAX, min_val = FLT_MAX;
            for (size_t i = 0; i < seq_dim; i++) {
                float v = src[i * hidden_dims + j];
                max_val = std::max(max_val, v);
                min_val = std::min(min_val, v);
            }
            scale[j] = (max_val - min_val) / 255.0f;
            if (scale[j] == 0.0f) scale[j] = 0.0001f;
            zp[j] = -min_val / scale[j];
        }
        for (size_t i = 0; i < seq_dim; i++) {
            for (size_t j = 0; j < hidden_dims; j++) {
                dst[i * dst_stride + j] =
                    static_cast<uint8_t>(std::round(std::max(src[i * hidden_dims + j] / scale[j] + zp[j], 0.0f)));
            }
        }
    }

    static void scalar_dequant_u8_by_channel(const uint8_t* src, float* dst,
                                             size_t seq_dim, size_t hidden_dims,
                                             size_t src_stride,
                                             const float* scale, const float* zp) {
        for (size_t i = 0; i < seq_dim; i++) {
            for (size_t j = 0; j < hidden_dims; j++) {
                dst[i * hidden_dims + j] = (static_cast<float>(src[i * src_stride + j]) - zp[j]) * scale[j];
            }
        }
    }

    static void scalar_quant_i8(const float* src, int8_t* dst, size_t n, float& scale) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < n; i++) {
            max_abs = std::max(max_abs, std::abs(src[i]));
        }
        scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 0.0001f;
        for (size_t i = 0; i < n; i++) {
            float tmp = std::round(src[i] / scale);
            tmp = std::max(tmp, -128.0f);
            tmp = std::min(tmp, 127.0f);
            dst[i] = static_cast<int8_t>(tmp);
        }
    }

    static void scalar_dequant_i8(const int8_t* src, float* dst, size_t n, float scale) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = static_cast<float>(src[i]) * scale;
        }
    }

    static uint8_t insert_u4(uint8_t dst_byte, uint8_t val, bool high_half) {
        uint8_t shift = high_half ? 0 : 4;
        return dst_byte | static_cast<uint8_t>(val << shift);
    }

    static uint8_t extract_u4(uint8_t byte, bool high_half) {
        uint8_t shift = high_half ? 0 : 4;
        return static_cast<uint8_t>((byte >> shift) & 0x0F);
    }

    static void scalar_quant_u4(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
        float max_val = -FLT_MAX, min_val = FLT_MAX;
        for (size_t i = 0; i < n; i++) {
            max_val = std::max(max_val, src[i]);
            min_val = std::min(min_val, src[i]);
        }
        scale = (max_val - min_val) / 15.0f;
        if (scale == 0.0f) scale = 0.0001f;
        zp = -min_val / scale;
        for (size_t i = 0; i < n; i++) {
            uint8_t val = static_cast<uint8_t>(std::min(15.0f, std::max(0.0f, std::round(src[i] / scale + zp))));
            uint8_t dst_val = (i % 2 == 0) ? 0 : dst[i / 2];
            dst[i / 2] = insert_u4(dst_val, val, static_cast<bool>(i % 2));
        }
    }

    static void scalar_dequant_u4(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
        for (size_t i = 0; i < n; i++) {
            uint8_t val = extract_u4(src[i / 2], static_cast<bool>(i % 2));
            dst[i] = (static_cast<float>(val) - zp) * scale;
        }
    }

    static void scalar_quant_u4_by_channel(const float* src, uint8_t* dst,
                                           size_t seq_dim, size_t hidden_dims,
                                           size_t dst_stride,
                                           float* scale, float* zp) {
        for (size_t j = 0; j < hidden_dims; j++) {
            float max_val = -FLT_MAX, min_val = FLT_MAX;
            for (size_t i = 0; i < seq_dim; i++) {
                float v = src[i * hidden_dims + j];
                max_val = std::max(max_val, v);
                min_val = std::min(min_val, v);
            }
            scale[j] = (max_val - min_val) / 15.0f;
            if (scale[j] == 0.0f) scale[j] = 0.0001f;
            zp[j] = -min_val / scale[j];
        }
        for (size_t i = 0; i < seq_dim; i++) {
            for (size_t j = 0; j < hidden_dims; j++) {
                uint8_t val = static_cast<uint8_t>(
                    std::min(15.0f, std::max(0.0f, std::round(src[i * hidden_dims + j] / scale[j] + zp[j]))));
                size_t byte_idx = i * dst_stride + j / 2;
                uint8_t dst_val = (j % 2 == 0) ? 0 : dst[byte_idx];
                dst[byte_idx] = insert_u4(dst_val, val, static_cast<bool>(j % 2));
            }
        }
    }

    static void scalar_dequant_u4_by_channel(const uint8_t* src, float* dst,
                                             size_t seq_dim, size_t hidden_dims,
                                             size_t src_stride,
                                             const float* scale, const float* zp) {
        for (size_t i = 0; i < seq_dim; i++) {
            for (size_t j = 0; j < hidden_dims; j++) {
                uint8_t val = extract_u4(src[i * src_stride + j / 2], static_cast<bool>(j % 2));
                dst[i * hidden_dims + j] = (static_cast<float>(val) - zp[j]) * scale[j];
            }
        }
    }

    // ── Generate float key data: [block][head][token][dim] layout ───────
    std::vector<float> generate_float_keys() const {
        std::vector<float> keys(kNumDataBlocks * kHeadNum * kBlockSize * kHeadSize);
        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < kHeadNum; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    const size_t global_token = block * kBlockSize + token;
                    for (size_t dim = 0; dim < kHeadSize; dim++) {
                        keys[((block * kHeadNum + head) * kBlockSize + token) * kHeadSize + dim] =
                            make_key_value(global_token, head * kHeadSize + dim);
                    }
                }
            }
        }
        return keys;
    }

    void fill_key_cache(ov::Tensor& key_cache_tensor) const {
        if (kKVCachePrecision == ov::element::f32) {
            fill_f32_key_cache(key_cache_tensor);
            return;
        }

        const auto float_keys = generate_float_keys();
        auto* raw = static_cast<uint8_t*>(key_cache_tensor.data());
        std::memset(raw, 0, key_cache_tensor.get_byte_size());

        if (kKVCachePrecision == ov::element::u8 && kQuantByChannel) {
            fill_u8_by_channel(raw, float_keys);
        } else if (kKVCachePrecision == ov::element::u8 && !kQuantByChannel) {
            fill_u8_by_token(raw, float_keys);
        } else if (kKVCachePrecision == ov::element::i8) {
            fill_i8_by_token(raw, float_keys);
        } else if (kKVCachePrecision == ov::element::u4 && kQuantByChannel) {
            fill_u4_by_channel(raw, float_keys);
        } else if (kKVCachePrecision == ov::element::u4 && !kQuantByChannel) {
            fill_u4_by_token(raw, float_keys);
        }
    }

    // dequant_keys layout: [num_heads, token_count, head_size]
    std::vector<float> dequant_key_cache(const ov::Tensor& key_cache_tensor) const {
        std::vector<float> dequant_keys(kHeadNum * kTokenCount * kHeadSize);

        if (kKVCachePrecision == ov::element::f32) {
            dequant_f32_key_cache(key_cache_tensor, dequant_keys);
        } else {
            const auto* raw = static_cast<const uint8_t*>(key_cache_tensor.data());
            if (kKVCachePrecision == ov::element::u8 && kQuantByChannel) {
                dequant_u8_by_channel(raw, dequant_keys);
            } else if (kKVCachePrecision == ov::element::u8 && !kQuantByChannel) {
                dequant_u8_by_token(raw, dequant_keys);
            } else if (kKVCachePrecision == ov::element::i8) {
                dequant_i8_by_token(raw, dequant_keys);
            } else if (kKVCachePrecision == ov::element::u4 && kQuantByChannel) {
                dequant_u4_by_channel(raw, dequant_keys);
            } else if (kKVCachePrecision == ov::element::u4 && !kQuantByChannel) {
                dequant_u4_by_token(raw, dequant_keys);
            }
        }

        return dequant_keys;
    }

    void fill_f32_key_cache(ov::Tensor& key_cache_tensor) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        auto* ptr = key_cache_tensor.data<float>();

        std::fill_n(ptr, key_cache_tensor.get_size(), 0.0f);
        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    const size_t global_token = block * kBlockSize + token;
                    for (size_t dim = 0; dim < S; dim++) {
                        const size_t offset = ((block * Hk + head) * kBlockSize + token) * S + dim;
                        ptr[offset] = make_key_value(global_token, head * S + dim);
                    }
                }
            }
        }
    }

    void dequant_f32_key_cache(const ov::Tensor& key_cache_tensor,
                               std::vector<float>& dequant_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const auto* ptr = key_cache_tensor.data<const float>();

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    const size_t global_token = block * kBlockSize + token;
                    for (size_t dim = 0; dim < S; dim++) {
                        const size_t src_offset = ((block * Hk + head) * kBlockSize + token) * S + dim;
                        dequant_keys[(head * kTokenCount + global_token) * S + dim] = ptr[src_offset];
                    }
                }
            }
        }
    }

    // ── u8 by-channel: cache shape [N, Hk, block_size + 8, S] element u8 ───
    void fill_u8_by_channel(uint8_t* raw, const std::vector<float>& float_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t params_size = 8;
        const size_t dim2 = kBlockSize + params_size;
        const size_t params_offset_bytes = 2 * sizeof(float) * S;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                uint8_t* base = raw + (block * Hk + head) * dim2 * S;
                auto* scales = reinterpret_cast<float*>(base);
                auto* zps = scales + S;
                uint8_t* data = base + params_offset_bytes;

                const float* src = float_keys.data() + ((block * Hk + head) * kBlockSize) * S;
                scalar_quant_u8_by_channel(src, data, kBlockSize, S, S, scales, zps);
            }
        }
    }

    void dequant_u8_by_channel(const uint8_t* raw, std::vector<float>& dequant_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t params_size = 8;
        const size_t dim2 = kBlockSize + params_size;
        const size_t params_offset_bytes = 2 * sizeof(float) * S;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                const uint8_t* base = raw + (block * Hk + head) * dim2 * S;
                const auto* scales = reinterpret_cast<const float*>(base);
                const auto* zps = scales + S;
                const uint8_t* data = base + params_offset_bytes;
                std::vector<float> dequant_block(kBlockSize * S);

                scalar_dequant_u8_by_channel(data, dequant_block.data(), kBlockSize, S, S, scales, zps);
                for (size_t token = 0; token < kBlockSize; token++) {
                    const size_t global_token = block * kBlockSize + token;
                    for (size_t dim = 0; dim < S; dim++) {
                        dequant_keys[(head * kTokenCount + global_token) * S + dim] =
                            dequant_block[token * S + dim];
                    }
                }
            }
        }
    }

    // ── u8 by-token: cache shape [N, Hk, block_size, S + 8] element u8 ───
    void fill_u8_by_token(uint8_t* raw, const std::vector<float>& float_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t adjusted_S = S + 8;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    uint8_t* token_base = raw + (((block * Hk + head) * kBlockSize + token) * adjusted_S);
                    auto* params = reinterpret_cast<float*>(token_base);
                    uint8_t* data = token_base + 2 * sizeof(float);

                    const float* src = float_keys.data() + ((block * Hk + head) * kBlockSize + token) * S;
                    float scale, zp;
                    scalar_quant_u8(src, data, S, scale, zp);
                    params[0] = scale;
                    params[1] = zp;
                }
            }
        }
    }

    void dequant_u8_by_token(const uint8_t* raw, std::vector<float>& dequant_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t adjusted_S = S + 8;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    const size_t global_token = block * kBlockSize + token;
                    const uint8_t* token_base = raw + (((block * Hk + head) * kBlockSize + token) * adjusted_S);
                    const auto* params = reinterpret_cast<const float*>(token_base);
                    const uint8_t* data = token_base + 2 * sizeof(float);

                    std::vector<float> dequant_token(S);
                    scalar_dequant_u8(data, dequant_token.data(), S, params[0], params[1]);
                    for (size_t dim = 0; dim < S; dim++) {
                        dequant_keys[(head * kTokenCount + global_token) * S + dim] = dequant_token[dim];
                    }
                }
            }
        }
    }

    // ── i8 by-token: cache shape [N, Hk, block_size, S + 4] element i8 ───
    void fill_i8_by_token(uint8_t* raw, const std::vector<float>& float_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t adjusted_S = S + 4;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    uint8_t* token_base = raw + (((block * Hk + head) * kBlockSize + token) * adjusted_S);
                    auto* scale_ptr = reinterpret_cast<float*>(token_base);
                    auto* data = reinterpret_cast<int8_t*>(token_base + sizeof(float));

                    const float* src = float_keys.data() + ((block * Hk + head) * kBlockSize + token) * S;
                    float scale;
                    scalar_quant_i8(src, data, S, scale);
                    *scale_ptr = scale;
                }
            }
        }
    }

    void dequant_i8_by_token(const uint8_t* raw, std::vector<float>& dequant_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t adjusted_S = S + 4;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    const size_t global_token = block * kBlockSize + token;
                    const uint8_t* token_base = raw + (((block * Hk + head) * kBlockSize + token) * adjusted_S);
                    const auto* scale_ptr = reinterpret_cast<const float*>(token_base);
                    const auto* data = reinterpret_cast<const int8_t*>(token_base + sizeof(float));

                    std::vector<float> dequant_token(S);
                    scalar_dequant_i8(data, dequant_token.data(), S, *scale_ptr);
                    for (size_t dim = 0; dim < S; dim++) {
                        dequant_keys[(head * kTokenCount + global_token) * S + dim] = dequant_token[dim];
                    }
                }
            }
        }
    }

    // ── u4 by-channel: cache shape [N, Hk, block_size + 16, S] element u4 ──
    void fill_u4_by_channel(uint8_t* raw, const std::vector<float>& float_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t params_size = 16;
        const size_t dim2 = kBlockSize + params_size;
        const size_t row_bytes = S / 2;
        const size_t params_offset_bytes = 2 * sizeof(float) * S;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                uint8_t* base = raw + (block * Hk + head) * dim2 * row_bytes;
                auto* scales = reinterpret_cast<float*>(base);
                auto* zps = scales + S;
                uint8_t* data = base + params_offset_bytes;

                const float* src = float_keys.data() + ((block * Hk + head) * kBlockSize) * S;
                scalar_quant_u4_by_channel(src, data, kBlockSize, S, row_bytes, scales, zps);
            }
        }
    }

    void dequant_u4_by_channel(const uint8_t* raw, std::vector<float>& dequant_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t params_size = 16;
        const size_t dim2 = kBlockSize + params_size;
        const size_t row_bytes = S / 2;
        const size_t params_offset_bytes = 2 * sizeof(float) * S;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                const uint8_t* base = raw + (block * Hk + head) * dim2 * row_bytes;
                const auto* scales = reinterpret_cast<const float*>(base);
                const auto* zps = scales + S;
                const uint8_t* data = base + params_offset_bytes;
                std::vector<float> dequant_block(kBlockSize * S);

                scalar_dequant_u4_by_channel(data, dequant_block.data(), kBlockSize, S, row_bytes, scales, zps);
                for (size_t token = 0; token < kBlockSize; token++) {
                    const size_t global_token = block * kBlockSize + token;
                    for (size_t dim = 0; dim < S; dim++) {
                        dequant_keys[(head * kTokenCount + global_token) * S + dim] =
                            dequant_block[token * S + dim];
                    }
                }
            }
        }
    }

    // ── u4 by-token: cache shape [N, Hk, block_size, S + 16] element u4 ──
    void fill_u4_by_token(uint8_t* raw, const std::vector<float>& float_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t adjusted_S_u4 = S + 16;
        const size_t row_bytes = adjusted_S_u4 / 2;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    uint8_t* token_base = raw +
                        (((block * Hk + head) * kBlockSize + token) * row_bytes);
                    auto* params = reinterpret_cast<float*>(token_base);
                    uint8_t* data = token_base + 2 * sizeof(float);

                    const float* src = float_keys.data() + ((block * Hk + head) * kBlockSize + token) * S;
                    float scale, zp;
                    scalar_quant_u4(src, data, S, scale, zp);
                    params[0] = scale;
                    params[1] = zp;
                }
            }
        }
    }

    void dequant_u4_by_token(const uint8_t* raw, std::vector<float>& dequant_keys) const {
        const size_t S = kHeadSize;
        const size_t Hk = kHeadNum;
        const size_t adjusted_S_u4 = S + 16;
        const size_t row_bytes = adjusted_S_u4 / 2;

        for (size_t block = 0; block < kNumDataBlocks; block++) {
            for (size_t head = 0; head < Hk; head++) {
                for (size_t token = 0; token < kBlockSize; token++) {
                    const size_t global_token = block * kBlockSize + token;
                    const uint8_t* token_base = raw + (((block * Hk + head) * kBlockSize + token) * row_bytes);
                    const auto* params = reinterpret_cast<const float*>(token_base);
                    const uint8_t* data = token_base + 2 * sizeof(float);

                    std::vector<float> dequant_token(S);
                    scalar_dequant_u4(data, dequant_token.data(), S, params[0], params[1]);
                    for (size_t dim = 0; dim < S; dim++) {
                        dequant_keys[(head * kTokenCount + global_token) * S + dim] = dequant_token[dim];
                    }
                }
            }
        }
    }

    // ── Build expected diversity from pre-computed key data ────────────────
    std::vector<float> build_expected_diversity(const std::vector<float>& key_data) const {
        ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(kStartSize, kEvictionSize, kBlockSize);
        auto block_diversity = calculator.calculate_block_diversity(
            key_data.data(), ov::Shape{kHeadNum, kTokenCount, kHeadSize});

        std::vector<float> flat;
        for (const auto& row : block_diversity) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return flat;
    }
};

TEST_P(PagedAttnAdaptiveRKVDiversityTest, smoke_AdaptiveRKVDiversityMatchesReference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    compile_model();
    auto infer_request = compiledModel.create_infer_request();

    // Calculate cache shapes/types based on quantization config
    auto calc_cache_info = [&](ov::element::Type prec, bool bychannel) {
        size_t extra_bs = 0, extra_hs = 0;
        if (prec == ov::element::u8) {
            if (bychannel) {
                extra_bs = 8;
            }
            else {
                extra_hs = 8;
            }
        } else if (prec == ov::element::i8) {
            extra_hs = 4;  // i8 is always by-token
        } else if (prec == ov::element::u4) {
            if (bychannel) {
                extra_bs = 16;
            } else {
                extra_hs = 16;
            }
        }
        ov::element::Type type = prec.is_integral() ? prec : ov::element::f32;
        ov::Shape shape{kNumBlocks, kHeadNum, kBlockSize + extra_bs, kHeadSize + extra_hs};
        return std::make_pair(type, shape);
    };

    auto [key_cache_type, key_cache_shape] = calc_cache_info(kKVCachePrecision, kQuantByChannel);
    // Value cache precision: i8 key cache uses SageAttn which sets value to u8
    ov::element::Type val_prec = (kKVCachePrecision == ov::element::i8) ? ov::element::u8 : kKVCachePrecision;
    auto [value_cache_type, value_cache_shape] = calc_cache_info(val_prec, false);  // value always by-token

    ov::Tensor q(ov::element::f32, {1, kHeadSize * kHeadNum});
    ov::Tensor k(ov::element::f32, {1, kHeadSize * kHeadNum});
    ov::Tensor v(ov::element::f32, {1, kHeadSize * kHeadNum});
    ov::Tensor key_cache(key_cache_type, key_cache_shape);
    ov::Tensor value_cache(value_cache_type, value_cache_shape);
    ov::Tensor past_lens(ov::element::i32, {1});
    ov::Tensor subsequence_begins(ov::element::i32, {2});
    ov::Tensor block_indices(ov::element::i32, {kNumBlocks});
    ov::Tensor block_indices_begins(ov::element::i32, {2});

    std::fill_n(q.data<float>(), kHeadSize * kHeadNum, 0.1f);
    std::fill_n(k.data<float>(), kHeadSize * kHeadNum, 0.2f);
    std::fill_n(v.data<float>(), kHeadSize * kHeadNum, 0.3f);
    std::memset(value_cache.data(), 0, value_cache.get_byte_size());

    past_lens.data<int32_t>()[0] = static_cast<int32_t>(kTokenCount);
    subsequence_begins.data<int32_t>()[0] = 0;
    subsequence_begins.data<int32_t>()[1] = 1;
    for (size_t i = 0; i < kNumBlocks; i++) {
        block_indices.data<int32_t>()[i] = static_cast<int32_t>(i);
    }
    block_indices_begins.data<int32_t>()[0] = 0;
    block_indices_begins.data<int32_t>()[1] = static_cast<int32_t>(kNumBlocks);

    fill_key_cache(key_cache);

    // Set tensors via compiled model inputs (shapes/types may differ from original model)
    for (const auto& input_port : compiledModel.inputs()) {
        for (const auto& name : input_port.get_names()) {
            if (name == "q") infer_request.set_tensor(input_port, q);
            else if (name == "k") infer_request.set_tensor(input_port, k);
            else if (name == "v") infer_request.set_tensor(input_port, v);
            else if (name == "key_cache.0") infer_request.set_tensor(input_port, key_cache);
            else if (name == "value_cache.0") infer_request.set_tensor(input_port, value_cache);
            else if (name == "past_lens") infer_request.set_tensor(input_port, past_lens);
            else if (name == "subsequence_begins") infer_request.set_tensor(input_port, subsequence_begins);
            else if (name == "block_indices") infer_request.set_tensor(input_port, block_indices);
            else if (name == "block_indices_begins") infer_request.set_tensor(input_port, block_indices_begins);
        }
    }

    infer_request.infer();
    const auto actual = infer_request.get_output_tensor(2);

    // Regarding to the reference implementation lacking u8/i8/u4 support,
    // we dequantize the key cache on the test side
    // and compare the diversity results against a float reference implementation.
    // This allows us to validate both the quantization and dequantization logic,
    // as well as the diversity calculation itself.
    const auto key_cache_float = dequant_key_cache(key_cache);
    const auto expected = build_expected_diversity(key_cache_float);

    ASSERT_EQ(actual.get_shape(), ov::Shape({expected.size()}));

    const float tolerance = kKVCachePrecision.is_integral() ? 1e-4f : 1e-5f;
    const auto* actual_ptr = actual.data<const float>();
    for (size_t idx = 0; idx < expected.size(); idx++) {
        EXPECT_NEAR(actual_ptr[idx], expected[idx], tolerance) << "Mismatch at index " << idx;
    }
}

const std::vector<DiversityTestParams> diversity_test_configs = {
    // f32 tests
    {4,  1, 32, 32, 32, 2, ov::element::f32, false, "single_head_minimal"},
    {16, 4, 32, 32, 32, 2, ov::element::f32, false, "multi_head_base"},
    {16, 4, 32, 32, 64, 3, ov::element::f32, false, "large_eviction"},
    {16, 2, 32, 64, 32, 3, ov::element::f32, false, "large_start"},
    {16, 4, 32, 64, 64, 4, ov::element::f32, false, "balanced_large"},
    // u8 by-channel — covers dequant<float, u8> + attn_dequant_by_channel_kernel<u8>
    {16, 4, 32, 32, 32, 2, ov::element::u8,  false, "u8_bychannel_base"},
    {16, 4, 32, 32, 64, 3, ov::element::u8,  false, "u8_bychannel_large_eviction"},
    // u8 by-token — covers dequant<float, u8> + attn_dequant_kernel<float, u8>
    {16, 4, 32, 32, 32, 2, ov::element::u8,  true,  "u8_bytoken_base"},
    {16, 4, 32, 32, 64, 3, ov::element::u8,  true,  "u8_bytoken_large_eviction"},
    // i8 by-token — covers i8-specific group-wise dequant path
    {16, 4, 32, 32, 32, 2, ov::element::i8,  false, "i8_bytoken_base"},
    {16, 4, 32, 32, 64, 3, ov::element::i8,  false, "i8_bytoken_large_eviction"},
    // u4 by-channel — covers dequant<float, u4> + attn_dequant_by_channel_kernel<u4>
    {16, 4, 32, 32, 32, 2, ov::element::u4,  false, "u4_bychannel_base"},
    {16, 4, 32, 32, 64, 3, ov::element::u4,  false, "u4_bychannel_large_eviction"},
    // u4 by-token — covers dequant<float, u4> + attn_dequant_kernel<float, u4>
    {16, 4, 32, 32, 32, 2, ov::element::u4,  true,  "u4_bytoken_base"},
    {16, 4, 32, 32, 64, 3, ov::element::u4,  true,  "u4_bytoken_large_eviction"},
};

INSTANTIATE_TEST_SUITE_P(smoke_AdaptiveRKVDiversity,
                         PagedAttnAdaptiveRKVDiversityTest,
                         ::testing::ValuesIn(diversity_test_configs),
                         PagedAttnAdaptiveRKVDiversityTest::getTestCaseName);

}  // namespace test
}  // namespace ov
