// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/quantization_utils.hpp"

namespace {

// Fixed small configuration exercised by this test: up to 3 independent sequences, each with its
// own past_len (kPastLens[s], including values that are NOT a multiple of key_block_size, to
// confirm the causal-tail derivation doesn't implicitly assume block-aligned history) and its own
// run of m_seq_blocks physical cache blocks, addressed through a non-identity BLOCK_INDICES permutation
// (see generate_inputs) so a kernel bug that walked physical blocks in logical/sequential order
// instead of resolving through BLOCK_INDICES would be caught. Unlike the dense SparseSDPA test
// (classes/sparse_sdpa.cpp, whose KEY/VALUE inputs carry the ENTIRE dense history), SparsePA's
// KEY/VALUE are documented as current-step-only -- history before past_len must already live in
// KEY_CACHE/VALUE_CACHE, so this test pre-populates those physical cache slots directly with
// synthetic "historical" data before invoking the op, exactly as a real multi-call decode/
// chunked-prefill sequence would have left them.
constexpr size_t kHq = 4;
constexpr size_t kHkv = 2;
constexpr size_t kDv = 16;
constexpr std::array<int32_t, 3> kPastLens = {0, 5, 13};
// Quantized caches use the 32-token pages ConvertPagedAttnInputs configures, so the history is longer: these past_len
// values start a sequence from scratch, continue inside the first page and, with 8 new tokens, cross into the second
// page (a by-channel key page requantized with new rows appended), and continue inside the second page.
constexpr std::array<int32_t, 3> kQuantPastLens = {0, 29, 45};
constexpr size_t kKmax = 4;

// Paged-cache layout: the page size (a runtime test parameter, see m_page_size; the op derives it
// from KEY_CACHE's dim 2) must be a multiple of key_block_size. The per-sequence physical block
// count (m_seq_blocks, computed in SetUp()) is sized so every kPastLens entry's worst case
// (kPastLens.back() + chunk_len tokens) comfortably fits regardless of the page size under test.
size_t seq_blocks_for(size_t page_size, size_t chunk_len, int32_t max_past_len) {
    return (static_cast<size_t>(max_past_len) + chunk_len + page_size - 1) / page_size;
}

void fill_pseudo_random(std::vector<float>& v, uint32_t seed_base) {
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t x = seed_base * 2654435761U + static_cast<uint32_t>(i) * 40503U + 1U;
        x ^= x >> 13;
        x *= 0x85EBCA6BU;
        x ^= x >> 16;
        v[i] = (static_cast<float>(x % 20000U) / 10000.0F) - 1.0F;
    }
}

// Independent (host-vector based) re-implementation of the sparse-pa.rst non-paged consumer
// spec, used purely as a test oracle -- NOT a copy of the CPU node's PlainTensor-based paged
// addressing. `key`/`value` here are the FULL per-sequence logical history (synthetic "historical"
// rows for position < past_len, followed by the current chunk's rows), already assembled by the
// caller (see calculate_refs) to mirror what the real KEY_CACHE/VALUE_CACHE contain once Q0 has
// run; `query` covers only the current chunk's chunk_len rows. `sel_indices`/`sel_count` are
// [chunk_len, num_selection_heads, Kmax] / [chunk_len, num_selection_heads], addressed by this
// sequence's own query row.
std::vector<float> run_sparse_pa_reference(size_t chunk_len,
                                           size_t h_sel,
                                           size_t k_head_size,
                                           size_t sel_block_size,
                                           float scale,
                                           int64_t past_len,
                                           const std::vector<float>& query,
                                           const std::vector<float>& key,
                                           const std::vector<float>& value,
                                           const std::vector<int32_t>& sel_indices,
                                           const std::vector<int32_t>& sel_count) {
    const int64_t history_len = past_len + static_cast<int64_t>(chunk_len);
    auto key_at = [&](size_t head, int64_t pos) {
        return &key[(head * static_cast<size_t>(history_len) + static_cast<size_t>(pos)) * k_head_size];
    };
    auto value_at = [&](size_t head, int64_t pos) {
        return &value[(head * static_cast<size_t>(history_len) + static_cast<size_t>(pos)) * kDv];
    };

    std::vector<float> output(chunk_len * kHq * kDv, 0.0F);
    const size_t group = kHq / kHkv;
    const size_t q_per_sel = kHq / h_sel;
    for (size_t l = 0; l < chunk_len; l++) {
        const int64_t token_pos = past_len + static_cast<int64_t>(l);
        for (size_t h = 0; h < kHq; h++) {
            const size_t kv_head = h / group;
            const size_t sh = h / q_per_sel;
            const float* q = &query[(l * kHq + h) * k_head_size];
            std::vector<float> acc(kDv, 0.0F);
            float running_max = -std::numeric_limits<float>::infinity();
            float running_sum = 0.0F;
            auto consume = [&](int64_t kpos) {
                if (kpos < 0 || kpos >= history_len) {
                    return;
                }
                if (kpos > token_pos) {
                    return;
                }
                const float* k = key_at(kv_head, kpos);
                const float* v = value_at(kv_head, kpos);
                float dot = 0.0F;
                for (size_t d = 0; d < k_head_size; d++) {
                    dot += q[d] * k[d];
                }
                const float score = dot * scale;
                const float new_max = std::max(running_max, score);
                const float corr = std::exp(running_max - new_max);
                const float p = std::exp(score - new_max);
                running_sum = running_sum * corr + p;
                for (size_t d = 0; d < kDv; d++) {
                    acc[d] = acc[d] * corr + p * v[d];
                }
                running_max = new_max;
            };

            const size_t nblocks = static_cast<size_t>(sel_count[(l * h_sel) + sh]);
            for (size_t i = 0; i < nblocks; i++) {
                const int64_t block_id = sel_indices[((l * h_sel) + sh) * kKmax + i];
                const int64_t block_start = block_id * static_cast<int64_t>(sel_block_size);
                for (int64_t p = block_start; p < block_start + static_cast<int64_t>(sel_block_size); p++) {
                    consume(p);
                }
            }

            // sel_indices/sel_count only ever carry *complete* selected blocks -- the current
            // (possibly incomplete) block is never in them. The consumer derives and attends that
            // tail itself from the query's own logical position.
            const int64_t tail_start =
                ((token_pos + 1) / static_cast<int64_t>(sel_block_size)) * static_cast<int64_t>(sel_block_size);
            for (int64_t p = tail_start; p <= token_pos; p++) {
                consume(p);
            }

            const float inv_sum = running_sum > 0.0F ? 1.0F / running_sum : 0.0F;
            for (size_t d = 0; d < kDv; d++) {
                output[(l * kHq + h) * kDv + d] = acc[d] * inv_sum;
            }
        }
    }
    return output;
}

}  // namespace

namespace ov {
namespace test {

std::string SparsePALayerCPUTest::getTestCaseName(const testing::TestParamInfo<SparsePALayerTestParams>& obj) {
    const auto& [device,
                 data_precision,
                 h_sel,
                 num_sequences,
                 k_head_size,
                 key_block_size,
                 page_size,
                 cache_precision,
                 key_bychannel,
                 group_size,
                 q_len] = obj.param;
    return "Target=" + device + "_DataPrecision=" + data_precision.get_type_name() + "_Hsel=" + std::to_string(h_sel) +
           "_NumSeq=" + std::to_string(num_sequences) + "_Dh=" + std::to_string(k_head_size) +
           "_Bk=" + std::to_string(key_block_size) + "_PageSize=" + std::to_string(page_size) +
           "_Cache=" + (cache_precision == ov::element::dynamic ? "plain" : cache_precision.get_type_name()) +
           "_KeyByChannel=" + std::to_string(key_bychannel) + "_Group=" + std::to_string(group_size) +
           "_QLen=" + std::to_string(q_len);
}

ov::Shape SparsePALayerCPUTest::cache_shape(bool is_key) const {
    const size_t head = is_key ? m_k_head_size : kDv;
    const size_t n_phys = m_num_sequences * m_seq_blocks;
    if (!quantized()) {
        return {n_phys, kHkv, m_page_size, head};
    }
    // Scale and zero point are f32; u4 counts them in half-byte elements.
    const size_t params = m_cache_precision == ov::element::u4 ? 16 : 8;
    const size_t group = m_group_size ? m_group_size : head;
    if (is_key && m_key_bychannel) {
        return {n_phys, kHkv, m_page_size + params, head};
    }
    return {n_phys, kHkv, m_page_size, head + params * (head / group)};
}

void SparsePALayerCPUTest::SetUp() {
    const auto& [device,
                 data_precision,
                 h_sel,
                 num_sequences,
                 k_head_size,
                 key_block_size,
                 page_size,
                 cache_precision,
                 key_bychannel,
                 group_size,
                 q_len] = GetParam();
    targetDevice = device;
    m_data_precision = data_precision;
    m_num_selection_heads = h_sel;
    m_num_sequences = num_sequences;
    m_k_head_size = k_head_size;
    m_key_block_size = key_block_size;
    m_page_size = page_size;
    m_cache_precision = cache_precision;
    m_key_bychannel = key_bychannel;
    m_group_size = group_size;
    m_q_len = q_len;
    m_past_lens = quantized() ? kQuantPastLens : kPastLens;
    m_seq_blocks = seq_blocks_for(m_page_size, m_q_len, m_past_lens.back());
    m_scale = 1.0F / std::sqrt(static_cast<float>(m_k_head_size));
    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    // ConvertPagedAttnInputs applies the KV cache precision (u8 by default) to the cache inputs: pin it to the data
    // precision for plain caches, or to the quantized precision, quantization mode and group size under test.
    if (quantized()) {
        configuration[ov::key_cache_precision.name()] = cache_precision;
        configuration[ov::value_cache_precision.name()] = cache_precision;
        configuration[ov::key_cache_group_size.name()] = static_cast<uint64_t>(group_size);
        configuration[ov::value_cache_group_size.name()] = static_cast<uint64_t>(group_size);
        configuration[ov::internal::key_cache_quant_mode.name()] =
            key_bychannel ? ov::internal::CacheQuantMode::BY_CHANNEL : ov::internal::CacheQuantMode::BY_TOKEN;
        configuration[ov::internal::value_cache_quant_mode.name()] = ov::internal::CacheQuantMode::BY_TOKEN;
    } else {
        configuration.insert(ov::hint::kv_cache_precision(data_precision));
    }
    configuration.insert(ov::hint::inference_precision(data_precision));
    if (data_precision == ov::element::bf16) {
        rel_threshold = 5e-2F;
        abs_threshold = 5e-2F;
    } else if (data_precision == ov::element::f16) {
        rel_threshold = 5e-3F;
        abs_threshold = 5e-3F;
    }

    const size_t chunk_total = m_num_sequences * m_q_len;
    const size_t n_phys = m_num_sequences * m_seq_blocks;

    auto p_query = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{chunk_total, kHq * m_k_head_size});
    auto p_key = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{chunk_total, kHkv * m_k_head_size});
    auto p_value = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{chunk_total, kHkv * kDv});
    // Quantized caches get their final element type and padded shape from ConvertPagedAttnInputs (driven by the head
    // runtime info set on the op below), so they start out unspecified.
    auto p_key_cache = quantized()
                           ? std::make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::PartialShape::dynamic(4))
                           : std::make_shared<ov::op::v0::Parameter>(data_precision, cache_shape(true));
    auto p_value_cache =
        quantized() ? std::make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::PartialShape::dynamic(4))
                    : std::make_shared<ov::op::v0::Parameter>(data_precision, cache_shape(false));
    auto p_past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences});
    auto p_subsequence_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    auto p_block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{n_phys});
    auto p_block_indices_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    auto p_sel_indices =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{chunk_total, m_num_selection_heads, kKmax});
    auto p_sel_count =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{chunk_total, m_num_selection_heads});

    // Input order follows ov::op::internal::PagedSparseAttention; SCALE and SINKS are constants
    // (SINKS is an empty tensor: sinks are not supported yet).
    ov::ParameterVector parameters{p_query,
                                   p_key,
                                   p_value,
                                   p_key_cache,
                                   p_value_cache,
                                   p_past_lens,
                                   p_subsequence_begins,
                                   p_block_indices,
                                   p_block_indices_begins,
                                   p_sel_indices,
                                   p_sel_count};
    std::vector<ov::Shape> param_shapes;
    for (const auto& p : parameters) {
        if (p == p_key_cache) {
            param_shapes.push_back(cache_shape(true));
        } else if (p == p_value_cache) {
            param_shapes.push_back(cache_shape(false));
        } else {
            param_shapes.push_back(p->get_shape());
        }
    }
    init_input_shapes(static_shapes_to_test_representation(param_shapes));

    auto scale = ov::op::v0::Constant::create(data_precision, ov::Shape{}, {m_scale});
    // Every floating-point input follows the data precision, which is also the inference precision.
    auto sinks = ov::op::v0::Constant::create(data_precision, ov::Shape{0}, std::vector<float>{});

    ov::OutputVector args = {p_query,
                             p_key,
                             p_value,
                             p_key_cache,
                             p_value_cache,
                             p_past_lens,
                             p_subsequence_begins,
                             p_block_indices,
                             p_block_indices_begins,
                             p_sel_indices,
                             p_sel_count,
                             scale,
                             sinks};
    auto sparse_pa = std::make_shared<ov::op::internal::PagedSparseAttention>(args, m_key_block_size);
    if (quantized()) {
        // Head metadata for ConvertPagedAttnInputs, which builds the padded quantized cache shapes from it.
        sparse_pa->get_rt_info()["num_k_heads"] = kHkv;
        sparse_pa->get_rt_info()["k_head_size"] = m_k_head_size;
        sparse_pa->get_rt_info()["num_v_heads"] = kHkv;
        sparse_pa->get_rt_info()["v_head_size"] = kDv;
    }

    function =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sparse_pa)}, parameters);
}

void SparsePALayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    const auto assign_precision = [this](ov::Tensor& t, const std::vector<float>& data) {
        if (m_data_precision == ov::element::bf16) {
            auto* dst = t.data<ov::bfloat16>();
            for (size_t i = 0; i < data.size(); i++) {
                dst[i] = ov::bfloat16(data[i]);
            }
        } else if (m_data_precision == ov::element::f16) {
            auto* dst = t.data<ov::float16>();
            for (size_t i = 0; i < data.size(); i++) {
                dst[i] = ov::float16(data[i]);
            }
        } else {
            std::copy(data.begin(), data.end(), t.data<float>());
        }
    };
    const auto make_zeros = [this](const ov::Shape& shape) {
        ov::Tensor t(m_data_precision, shape);
        if (m_data_precision == ov::element::bf16) {
            std::fill_n(t.data<ov::bfloat16>(), t.get_size(), ov::bfloat16(0.0F));
        } else if (m_data_precision == ov::element::f16) {
            std::fill_n(t.data<ov::float16>(), t.get_size(), ov::float16(0.0F));
        } else {
            std::fill_n(t.data<float>(), t.get_size(), 0.0F);
        }
        return t;
    };
    const auto make_i32 = [](const std::vector<int32_t>& values) {
        ov::Tensor t(ov::element::i32, ov::Shape{values.size()});
        std::copy(values.begin(), values.end(), t.data<int32_t>());
        return t;
    };

    const size_t n_phys = m_num_sequences * m_seq_blocks;
    const size_t chunk_total = m_num_sequences * m_q_len;

    // Non-identity permutation (simple reversal) across all physical blocks shared by every
    // sequence, so a kernel bug that walked physical blocks in logical/sequential order instead
    // of resolving through block_indices would produce numerically wrong results.
    std::vector<int32_t> block_indices(n_phys);
    for (size_t i = 0; i < n_phys; i++) {
        block_indices[i] = static_cast<int32_t>(n_phys - 1 - i);
    }
    // Mirrors the kernel's own token_phys() addressing (sparse_pa.cpp) exactly -- needed both to
    // pre-populate KEY_CACHE/VALUE_CACHE with synthetic historical data at the correct physical
    // slot before the op runs, and (in calculate_refs) to read that same data back for the oracle.
    const auto token_phys = [&](size_t seq, int64_t logical_pos) {
        const int64_t block_begin = static_cast<int64_t>(seq * m_seq_blocks);
        const int64_t phys_idx = block_begin + logical_pos / static_cast<int64_t>(m_page_size);
        const size_t phys_block = static_cast<size_t>(block_indices[static_cast<size_t>(phys_idx)]);
        const size_t slot = static_cast<size_t>(logical_pos % static_cast<int64_t>(m_page_size));
        return std::make_pair(phys_block, slot);
    };

    ov::Tensor query(m_data_precision, ov::Shape{chunk_total, kHq * m_k_head_size});
    ov::Tensor key(m_data_precision, ov::Shape{chunk_total, kHkv * m_k_head_size});
    ov::Tensor value(m_data_precision, ov::Shape{chunk_total, kHkv * kDv});
    const auto make_cache = [&](bool is_key) {
        if (!quantized()) {
            return make_zeros(cache_shape(is_key));
        }
        ov::Tensor t(m_cache_precision, cache_shape(is_key));
        std::memset(t.data(), 0, t.get_byte_size());
        return t;
    };
    ov::Tensor key_cache = make_cache(true);
    ov::Tensor value_cache = make_cache(false);
    {
        std::vector<float> query_data(query.get_size());
        std::vector<float> key_data(key.get_size());
        std::vector<float> value_data(value.get_size());
        for (size_t s = 0; s < m_num_sequences; s++) {
            std::vector<float> q_slot(m_q_len * kHq * m_k_head_size);
            std::vector<float> k_slot(m_q_len * kHkv * m_k_head_size);
            std::vector<float> v_slot(m_q_len * kHkv * kDv);
            fill_pseudo_random(q_slot, 1 + static_cast<uint32_t>(s) * 11U);
            fill_pseudo_random(k_slot, 2 + static_cast<uint32_t>(s) * 11U);
            fill_pseudo_random(v_slot, 3 + static_cast<uint32_t>(s) * 11U);
            std::copy(q_slot.begin(), q_slot.end(), query_data.begin() + static_cast<ptrdiff_t>(s * q_slot.size()));
            std::copy(k_slot.begin(), k_slot.end(), key_data.begin() + static_cast<ptrdiff_t>(s * k_slot.size()));
            std::copy(v_slot.begin(), v_slot.end(), value_data.begin() + static_cast<ptrdiff_t>(s * v_slot.size()));
        }
        assign_precision(query, query_data);
        assign_precision(key, key_data);
        assign_precision(value, value_data);
    }

    // Writes the "historical" rows (positions < past_len) of every sequence into the quantized caches, in the layout
    // PagedAttention uses, and records their dequantized values for the oracle (m_hist_key / m_hist_value, laid out
    // [kv, past_len, head_size]). By-channel key pages hold the scale/zero-point of the history rows of the page.
    const auto populate_quantized_history = [&]() {
        const bool u4 = m_cache_precision == ov::element::u4;
        const size_t sub_byte = u4 ? 2 : 1;
        const size_t params_bytes = 2 * sizeof(float);
        m_hist_key.assign(m_num_sequences, {});
        m_hist_value.assign(m_num_sequences, {});
        auto* key_raw = static_cast<uint8_t*>(key_cache.data());
        auto* value_raw = static_cast<uint8_t*>(value_cache.data());
        const auto key_dims = key_cache.get_shape();
        const auto value_dims = value_cache.get_shape();
        const auto quant_row = [&](const float* src, uint8_t* row, size_t D) {
            const size_t group = m_group_size ? m_group_size : D;
            size_t offset = 0;
            for (size_t g = 0; g < D; g += group) {
                float scale = 0.0F;
                float zp = 0.0F;
                if (u4) {
                    CPUTestUtils::scalar_quant_u4(src + g, row + offset + params_bytes, group, scale, zp);
                } else {
                    CPUTestUtils::scalar_quant_u8(src + g, row + offset + params_bytes, group, scale, zp);
                }
                auto* params = reinterpret_cast<float*>(row + offset);
                params[0] = scale;
                params[1] = zp;
                offset += params_bytes + group / sub_byte;
            }
        };
        const auto dequant_row = [&](const uint8_t* row, float* dst, size_t D) {
            const size_t group = m_group_size ? m_group_size : D;
            size_t offset = 0;
            for (size_t g = 0; g < D; g += group) {
                const auto* params = reinterpret_cast<const float*>(row + offset);
                if (u4) {
                    CPUTestUtils::scalar_dequant_u4(row + offset + params_bytes, dst + g, group, params[0], params[1]);
                } else {
                    CPUTestUtils::scalar_dequant_u8(row + offset + params_bytes, dst + g, group, params[0], params[1]);
                }
                offset += params_bytes + group / sub_byte;
            }
        };
        for (size_t s = 0; s < m_num_sequences; s++) {
            const auto past_len = static_cast<size_t>(m_past_lens[s]);
            std::vector<float> k_float(kHkv * past_len * m_k_head_size);
            std::vector<float> v_float(kHkv * past_len * kDv);
            for (size_t pos = 0; pos < past_len; pos++) {
                for (size_t kv = 0; kv < kHkv; kv++) {
                    const uint32_t seed = 1000 + static_cast<uint32_t>(s) * 97U + static_cast<uint32_t>(pos) * 7U +
                                          static_cast<uint32_t>(kv) * 3U;
                    std::vector<float> k_row(m_k_head_size);
                    std::vector<float> v_row(kDv);
                    fill_pseudo_random(k_row, seed);
                    fill_pseudo_random(v_row, seed + 500U);
                    std::copy(k_row.begin(), k_row.end(), k_float.begin() + (kv * past_len + pos) * m_k_head_size);
                    std::copy(v_row.begin(), v_row.end(), v_float.begin() + (kv * past_len + pos) * kDv);
                }
            }
            m_hist_key[s].assign(k_float.size(), 0.0F);
            m_hist_value[s].assign(v_float.size(), 0.0F);
            for (size_t pos = 0; pos < past_len; pos++) {
                const auto [phys_block, slot] = token_phys(s, static_cast<int64_t>(pos));
                for (size_t kv = 0; kv < kHkv; kv++) {
                    const size_t idx = kv * past_len + pos;
                    // Value (and by-token key) rows: [phys_block, kv, slot] of the padded cache.
                    const size_t v_row_bytes = value_dims[3] / sub_byte;
                    auto* v_row = value_raw + ((phys_block * kHkv + kv) * m_page_size + slot) * v_row_bytes;
                    quant_row(&v_float[idx * kDv], v_row, kDv);
                    dequant_row(v_row, &m_hist_value[s][idx * kDv], kDv);
                    if (!m_key_bychannel) {
                        const size_t k_row_bytes = key_dims[3] / sub_byte;
                        auto* k_row = key_raw + ((phys_block * kHkv + kv) * m_page_size + slot) * k_row_bytes;
                        quant_row(&k_float[idx * m_k_head_size], k_row, m_k_head_size);
                        dequant_row(k_row, &m_hist_key[s][idx * m_k_head_size], m_k_head_size);
                    }
                }
            }
            if (m_key_bychannel) {
                // One scale/zero-point per channel and page, computed over the history rows of the page.
                const size_t row_bytes = m_k_head_size / sub_byte;
                for (size_t kv = 0; kv < kHkv; kv++) {
                    for (size_t page_start = 0; page_start < past_len; page_start += m_page_size) {
                        const size_t rows = std::min(m_page_size, past_len - page_start);
                        const auto [phys_block, slot0] = token_phys(s, static_cast<int64_t>(page_start));
                        auto* base = key_raw + (phys_block * kHkv + kv) * key_dims[2] * row_bytes;
                        auto* scales = reinterpret_cast<float*>(base);
                        float* zps = scales + m_k_head_size;
                        uint8_t* data = base + params_bytes * m_k_head_size;
                        const float* src = &k_float[(kv * past_len + page_start) * m_k_head_size];
                        float* dst = &m_hist_key[s][(kv * past_len + page_start) * m_k_head_size];
                        if (u4) {
                            CPUTestUtils::scalar_quant_u4_by_channel(src,
                                                                     data,
                                                                     rows,
                                                                     m_k_head_size,
                                                                     row_bytes,
                                                                     scales,
                                                                     zps);
                            CPUTestUtils::scalar_dequant_u4_by_channel(data,
                                                                       dst,
                                                                       rows,
                                                                       m_k_head_size,
                                                                       row_bytes,
                                                                       scales,
                                                                       zps);
                        } else {
                            CPUTestUtils::scalar_quant_u8_by_channel(src,
                                                                     data,
                                                                     rows,
                                                                     m_k_head_size,
                                                                     row_bytes,
                                                                     scales,
                                                                     zps);
                            CPUTestUtils::scalar_dequant_u8_by_channel(data,
                                                                       dst,
                                                                       rows,
                                                                       m_k_head_size,
                                                                       row_bytes,
                                                                       scales,
                                                                       zps);
                        }
                        (void)slot0;
                    }
                }
            }
        }
    };

    if (quantized()) {
        populate_quantized_history();
    } else {
        // Pre-populate KEY_CACHE/VALUE_CACHE with synthetic "historical" data for every logical
        // position strictly before this sequence's past_len, placed at the exact physical (block,
        // slot) address the op's own Q0 addressing would also compute -- exactly what a real
        // multi-call decode/chunked-prefill sequence would have already written on a prior call.
        for (size_t s = 0; s < m_num_sequences; s++) {
            const auto past_len = static_cast<int64_t>(m_past_lens[s]);
            for (int64_t pos = 0; pos < past_len; pos++) {
                const auto [phys_block, slot] = token_phys(s, pos);
                for (size_t kv = 0; kv < kHkv; kv++) {
                    std::vector<float> k_row(m_k_head_size);
                    std::vector<float> v_row(kDv);
                    const uint32_t seed = 1000 + static_cast<uint32_t>(s) * 97U + static_cast<uint32_t>(pos) * 7U +
                                          static_cast<uint32_t>(kv) * 3U;
                    fill_pseudo_random(k_row, seed);
                    fill_pseudo_random(v_row, seed + 500U);
                    if (m_data_precision == ov::element::bf16) {
                        auto* kdst = key_cache.data<ov::bfloat16>() +
                                     ((phys_block * kHkv + kv) * m_page_size + slot) * m_k_head_size;
                        auto* vdst =
                            value_cache.data<ov::bfloat16>() + ((phys_block * kHkv + kv) * m_page_size + slot) * kDv;
                        for (size_t d = 0; d < m_k_head_size; d++) {
                            kdst[d] = ov::bfloat16(k_row[d]);
                        }
                        for (size_t d = 0; d < kDv; d++) {
                            vdst[d] = ov::bfloat16(v_row[d]);
                        }
                    } else if (m_data_precision == ov::element::f16) {
                        auto* kdst = key_cache.data<ov::float16>() +
                                     ((phys_block * kHkv + kv) * m_page_size + slot) * m_k_head_size;
                        auto* vdst =
                            value_cache.data<ov::float16>() + ((phys_block * kHkv + kv) * m_page_size + slot) * kDv;
                        for (size_t d = 0; d < m_k_head_size; d++) {
                            kdst[d] = ov::float16(k_row[d]);
                        }
                        for (size_t d = 0; d < kDv; d++) {
                            vdst[d] = ov::float16(v_row[d]);
                        }
                    } else {
                        auto* kdst =
                            key_cache.data<float>() + ((phys_block * kHkv + kv) * m_page_size + slot) * m_k_head_size;
                        auto* vdst = value_cache.data<float>() + ((phys_block * kHkv + kv) * m_page_size + slot) * kDv;
                        std::copy(k_row.begin(), k_row.end(), kdst);
                        std::copy(v_row.begin(), v_row.end(), vdst);
                    }
                }
            }
        }
    }

    // Synthetic selection: the up-to-kKmax *complete* blocks immediately strictly before the
    // query's own block (in the sequence's full logical history, including the pre-populated
    // historical span), in ascending order -- per the documented producer contract, the current
    // (possibly incomplete) block must never appear in sel_indices/sel_count, and sel_indices must
    // be strictly ascending and duplicate-free; the consumer derives and attends the current block
    // automatically instead.
    std::vector<int32_t> sel_indices_flat(chunk_total * m_num_selection_heads * kKmax, -1);
    std::vector<int32_t> sel_count_flat(chunk_total * m_num_selection_heads, 0);
    for (size_t s = 0; s < m_num_sequences; s++) {
        const auto past_len = static_cast<int64_t>(m_past_lens[s]);
        for (size_t l = 0; l < m_q_len; l++) {
            const size_t token = s * m_q_len + l;
            const int64_t token_pos = past_len + static_cast<int64_t>(l);
            const auto cur_block = static_cast<int32_t>(token_pos / static_cast<int64_t>(m_key_block_size));
            const size_t num_sel = std::min(static_cast<size_t>(cur_block), kKmax);
            for (size_t sh = 0; sh < m_num_selection_heads; sh++) {
                const size_t base = token * m_num_selection_heads + sh;
                for (size_t k = 0; k < num_sel; k++) {
                    sel_indices_flat[base * kKmax + k] =
                        cur_block - static_cast<int32_t>(num_sel) + static_cast<int32_t>(k);
                }
                sel_count_flat[base] = static_cast<int32_t>(num_sel);
            }
        }
    }

    // Tensors follow the parameter order from SetUp(): the op's inputs 0..10 (SCALE/SINKS are
    // constants), then the optional QB_START.
    const auto& params = function->get_parameters();
    std::vector<ov::Tensor> tensors(params.size());
    tensors[0] = query;
    tensors[1] = key;
    tensors[2] = value;
    tensors[3] = key_cache;
    tensors[4] = value_cache;
    {
        std::vector<int32_t> past_lens_values(m_num_sequences);
        for (size_t s = 0; s < m_num_sequences; s++) {
            past_lens_values[s] = m_past_lens[s];
        }
        tensors[5] = make_i32(past_lens_values);
    }
    std::vector<int32_t> subsequence_begins(m_num_sequences + 1);
    for (size_t s = 0; s <= m_num_sequences; s++) {
        subsequence_begins[s] = static_cast<int32_t>(s * m_q_len);
    }
    tensors[6] = make_i32(subsequence_begins);
    tensors[7] = make_i32(block_indices);
    {
        std::vector<int32_t> block_indices_begins(m_num_sequences + 1);
        for (size_t s = 0; s <= m_num_sequences; s++) {
            block_indices_begins[s] = static_cast<int32_t>(s * m_seq_blocks);
        }
        tensors[8] = make_i32(block_indices_begins);
    }
    {
        ov::Tensor t(ov::element::i32, ov::Shape{chunk_total, m_num_selection_heads, kKmax});
        std::copy(sel_indices_flat.begin(), sel_indices_flat.end(), t.data<int32_t>());
        tensors[9] = t;
    }
    {
        ov::Tensor t(ov::element::i32, ov::Shape{chunk_total, m_num_selection_heads});
        std::copy(sel_count_flat.begin(), sel_count_flat.end(), t.data<int32_t>());
        tensors[10] = t;
    }

    for (size_t i = 0; i < params.size(); i++) {
        host_inputs[params[i]] = tensors[i];
        inputs[params[i]] = tensors[i];
    }
}

std::vector<ov::Tensor> SparsePALayerCPUTest::calculate_refs() {
    const auto& params = function->get_parameters();
    const auto to_f32 = [&](const std::shared_ptr<ov::Node>& p) {
        const auto& t = host_inputs.at(p);
        std::vector<float> out(t.get_size());
        if (t.get_element_type() == ov::element::bf16) {
            const auto* ptr = t.data<const ov::bfloat16>();
            std::copy(ptr, ptr + t.get_size(), out.begin());
        } else if (t.get_element_type() == ov::element::f16) {
            const auto* ptr = t.data<const ov::float16>();
            std::copy(ptr, ptr + t.get_size(), out.begin());
        } else {
            const auto* ptr = t.data<const float>();
            std::copy(ptr, ptr + t.get_size(), out.begin());
        }
        return out;
    };
    const auto to_i32 = [&](const std::shared_ptr<ov::Node>& p) {
        const auto& t = host_inputs.at(p);
        const auto* ptr = t.data<const int32_t>();
        return std::vector<int32_t>(ptr, ptr + t.get_size());
    };

    auto query = to_f32(params[0]);
    auto key = to_f32(params[1]);
    auto value = to_f32(params[2]);
    // Quantized caches are raw bytes; the oracle works from m_hist_key / m_hist_value instead.
    auto key_cache = quantized() ? std::vector<float>{} : to_f32(params[3]);
    auto value_cache = quantized() ? std::vector<float>{} : to_f32(params[4]);
    auto block_indices = to_i32(params[7]);
    auto sel_indices = to_i32(params[9]);
    auto sel_count = to_i32(params[10]);

    const auto token_phys = [&](size_t seq, int64_t logical_pos) {
        const int64_t block_begin = static_cast<int64_t>(seq * m_seq_blocks);
        const int64_t phys_idx = block_begin + logical_pos / static_cast<int64_t>(m_page_size);
        const size_t phys_block = static_cast<size_t>(block_indices[static_cast<size_t>(phys_idx)]);
        const size_t slot = static_cast<size_t>(logical_pos % static_cast<int64_t>(m_page_size));
        return std::make_pair(phys_block, slot);
    };

    const size_t chunk_total = m_num_sequences * m_q_len;
    ov::Tensor output(ov::element::f32, ov::Shape{chunk_total, kHq * kDv});
    auto* output_data = output.data<float>();

    for (size_t s = 0; s < m_num_sequences; s++) {
        const auto past_len = static_cast<int64_t>(m_past_lens[s]);
        const int64_t history_len = past_len + static_cast<int64_t>(m_q_len);

        // Reassemble this sequence's full logical K/V history: historical rows (< past_len) are
        // read back out of the pre-populated KEY_CACHE/VALUE_CACHE through the same physical
        // addressing the op itself uses; current-chunk rows (>= past_len) are exactly this
        // sequence's slice of the KEY/VALUE test inputs (what the op's own Q0 stage would have
        // just written into the cache).
        std::vector<float> history_key(kHkv * static_cast<size_t>(history_len) * m_k_head_size);
        std::vector<float> history_value(kHkv * static_cast<size_t>(history_len) * kDv);
        if (!quantized()) {
            for (int64_t pos = 0; pos < history_len; pos++) {
                const auto [phys_block, slot] = token_phys(s, pos);
                for (size_t kv = 0; kv < kHkv; kv++) {
                    float* k_dst = &history_key[(kv * static_cast<size_t>(history_len) + static_cast<size_t>(pos)) *
                                                m_k_head_size];
                    float* v_dst =
                        &history_value[(kv * static_cast<size_t>(history_len) + static_cast<size_t>(pos)) * kDv];
                    if (pos < past_len) {
                        const float* k_src =
                            &key_cache[((phys_block * kHkv + kv) * m_page_size + slot) * m_k_head_size];
                        const float* v_src = &value_cache[((phys_block * kHkv + kv) * m_page_size + slot) * kDv];
                        std::copy_n(k_src, m_k_head_size, k_dst);
                        std::copy_n(v_src, kDv, v_dst);
                    } else {
                        const size_t token = s * m_q_len + static_cast<size_t>(pos - past_len);
                        const float* k_src = &key[(token * kHkv + kv) * m_k_head_size];
                        const float* v_src = &value[(token * kHkv + kv) * kDv];
                        std::copy_n(k_src, m_k_head_size, k_dst);
                        std::copy_n(v_src, kDv, v_dst);
                    }
                }
            }
        } else {
            // Quantized caches: history rows are the dequantized rows written by generate_inputs(); the op then
            // quantizes the current rows the same way (by token), or requantizes every touched by-channel key page
            // from its dequantized history rows plus the new rows, so the oracle emulates both steps.
            const bool u4 = m_cache_precision == ov::element::u4;
            const size_t D = m_k_head_size;
            const auto past = static_cast<size_t>(past_len);
            const auto total = static_cast<size_t>(history_len);
            const auto new_row = [&](bool is_key, size_t pos, size_t kv) {
                const size_t token = s * m_q_len + (pos - past);
                return is_key ? &key[(token * kHkv + kv) * D] : &value[(token * kHkv + kv) * kDv];
            };
            const auto fake_quant_token = [&](const float* src, float* dst, size_t head) {
                const size_t group = m_group_size ? m_group_size : head;
                std::vector<uint8_t> packed(group);
                for (size_t g = 0; g < head; g += group) {
                    float scale = 0.0F;
                    float zp = 0.0F;
                    if (u4) {
                        CPUTestUtils::scalar_quant_u4(src + g, packed.data(), group, scale, zp);
                        CPUTestUtils::scalar_dequant_u4(packed.data(), dst + g, group, scale, zp);
                    } else {
                        CPUTestUtils::scalar_quant_u8(src + g, packed.data(), group, scale, zp);
                        CPUTestUtils::scalar_dequant_u8(packed.data(), dst + g, group, scale, zp);
                    }
                }
            };
            for (size_t kv = 0; kv < kHkv; kv++) {
                for (size_t pos = 0; pos < total; pos++) {
                    float* v_dst = &history_value[(kv * total + pos) * kDv];
                    if (pos < past) {
                        std::copy_n(&m_hist_value[s][(kv * past + pos) * kDv], kDv, v_dst);
                    } else {
                        fake_quant_token(new_row(false, pos, kv), v_dst, kDv);
                    }
                    if (!m_key_bychannel) {
                        float* k_dst = &history_key[(kv * total + pos) * D];
                        if (pos < past) {
                            std::copy_n(&m_hist_key[s][(kv * past + pos) * D], D, k_dst);
                        } else {
                            fake_quant_token(new_row(true, pos, kv), k_dst, D);
                        }
                    }
                }
                if (!m_key_bychannel) {
                    continue;
                }
                const size_t row_bytes = u4 ? D / 2 : D;
                for (size_t page_start = 0; page_start < total; page_start += m_page_size) {
                    const size_t rows = std::min(m_page_size, total - page_start);
                    float* dst = &history_key[(kv * total + page_start) * D];
                    if (page_start + rows <= past) {
                        // Page untouched by this call: only the history quantization applies.
                        std::copy_n(&m_hist_key[s][(kv * past + page_start) * D], rows * D, dst);
                        continue;
                    }
                    std::vector<float> page_rows(rows * D);
                    for (size_t r = 0; r < rows; r++) {
                        const size_t pos = page_start + r;
                        const float* src = pos < past ? &m_hist_key[s][(kv * past + pos) * D] : new_row(true, pos, kv);
                        std::copy_n(src, D, &page_rows[r * D]);
                    }
                    std::vector<uint8_t> data(rows * row_bytes, 0);
                    std::vector<float> scales(D);
                    std::vector<float> zps(D);
                    if (u4) {
                        CPUTestUtils::scalar_quant_u4_by_channel(page_rows.data(),
                                                                 data.data(),
                                                                 rows,
                                                                 D,
                                                                 row_bytes,
                                                                 scales.data(),
                                                                 zps.data());
                        CPUTestUtils::scalar_dequant_u4_by_channel(data.data(),
                                                                   dst,
                                                                   rows,
                                                                   D,
                                                                   row_bytes,
                                                                   scales.data(),
                                                                   zps.data());
                    } else {
                        CPUTestUtils::scalar_quant_u8_by_channel(page_rows.data(),
                                                                 data.data(),
                                                                 rows,
                                                                 D,
                                                                 row_bytes,
                                                                 scales.data(),
                                                                 zps.data());
                        CPUTestUtils::scalar_dequant_u8_by_channel(data.data(),
                                                                   dst,
                                                                   rows,
                                                                   D,
                                                                   row_bytes,
                                                                   scales.data(),
                                                                   zps.data());
                    }
                }
            }
        }

        std::vector<float> query_s(query.begin() + static_cast<ptrdiff_t>(s * m_q_len * kHq * m_k_head_size),
                                   query.begin() + static_cast<ptrdiff_t>((s + 1) * m_q_len * kHq * m_k_head_size));
        std::vector<int32_t> sel_indices_s(
            sel_indices.begin() + static_cast<ptrdiff_t>(s * m_q_len * m_num_selection_heads * kKmax),
            sel_indices.begin() + static_cast<ptrdiff_t>((s + 1) * m_q_len * m_num_selection_heads * kKmax));
        std::vector<int32_t> sel_count_s(
            sel_count.begin() + static_cast<ptrdiff_t>(s * m_q_len * m_num_selection_heads),
            sel_count.begin() + static_cast<ptrdiff_t>((s + 1) * m_q_len * m_num_selection_heads));

        auto out_s = run_sparse_pa_reference(m_q_len,
                                             m_num_selection_heads,
                                             m_k_head_size,
                                             m_key_block_size,
                                             m_scale,
                                             past_len,
                                             query_s,
                                             history_key,
                                             history_value,
                                             sel_indices_s,
                                             sel_count_s);
        std::copy(out_s.begin(), out_s.end(), output_data + s * m_q_len * kHq * kDv);
    }
    return {output};
}

void SparsePALayerCPUTest::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

TEST_P(SparsePALayerCPUTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov