// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_constants.hpp"
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

void fill_pseudo_random(std::vector<float>& v, uint32_t seed_base) {
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t x = seed_base * 2654435761U + static_cast<uint32_t>(i) * 40503U + 1U;
        x ^= x >> 13;
        x *= 0x85EBCA6BU;
        x ^= x >> 16;
        v[i] = (static_cast<float>(x % 20000U) / 10000.0F) - 1.0F;
    }
}

uint32_t next_random(uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return state >> 8;
}

// Synthetic indexer output for one (token, selection head): `num_complete` = (position + 1) / key_block_size blocks
// are complete at the token, so any of them may be selected -- including the one that ends exactly at the token, which
// leaves the causal tail empty. Entries after the count are -1.
void make_selection(ov::test::SparsePASelection mode,
                    size_t max_blocks,
                    size_t num_complete,
                    uint32_t& rng,
                    int32_t* indices,
                    int32_t& count) {
    std::fill(indices, indices + max_blocks, -1);
    count = 0;
    if (mode == ov::test::SparsePASelection::kEmpty || num_complete == 0) {
        return;
    }
    const size_t limit = std::min(num_complete, max_blocks);
    size_t n = limit;
    if (mode == ov::test::SparsePASelection::kRecent) {
        for (size_t k = 0; k < n; k++) {
            indices[k] = static_cast<int32_t>(num_complete - n + k);
        }
    } else {
        n = next_random(rng) % (limit + 1);
        std::vector<int32_t> pool(num_complete);
        std::iota(pool.begin(), pool.end(), 0);
        for (size_t k = 0; k < n; k++) {
            const size_t j = k + next_random(rng) % (num_complete - k);
            std::swap(pool[k], pool[j]);
        }
        std::sort(pool.begin(), pool.begin() + static_cast<ptrdiff_t>(n));
        std::copy_n(pool.begin(), n, indices);
    }
    count = static_cast<int32_t>(n);
}

// Quantize + dequantize one by-token group with the plugin's formulas (attn_quant_kernel.hpp): min/max affine
// quantization, evaluated like its vector code as fma(x, 1 / scale, zp) rounded to nearest even. Evaluating x / scale +
// zp with std::round instead differs by one level on values within an ulp of a rounding tie.
void fake_quant_group(bool u4, const float* src, float* dst, size_t n) {
    float max_val = -std::numeric_limits<float>::max();
    float min_val = std::numeric_limits<float>::max();
    for (size_t i = 0; i < n; i++) {
        max_val = std::max(max_val, src[i]);
        min_val = std::min(min_val, src[i]);
    }
    const float levels = u4 ? 15.0F : 255.0F;
    float scale = (max_val - min_val) / levels;
    if (scale == 0.0F) {
        scale = 0.0001F;
    }
    const float zp = -min_val / scale;
    const float inv_scale = 1.0F / scale;
    for (size_t i = 0; i < n; i++) {
        const float q = std::min(levels, std::max(0.0F, std::nearbyint(std::fma(src[i], inv_scale, zp))));
        dst[i] = (q - zp) * scale;
    }
}

void fake_quant_by_token(ov::element::Type prec, size_t group_size, const float* src, float* dst, size_t head) {
    const size_t group = group_size ? group_size : head;
    for (size_t g = 0; g < head; g += group) {
        fake_quant_group(prec == ov::element::u4, src + g, dst + g, group);
    }
}

// One scale and zero point per channel over `rows` rows of a key page.
void fake_quant_by_channel(ov::element::Type prec, const float* src, float* dst, size_t rows, size_t hidden) {
    const bool u4 = prec == ov::element::u4;
    const size_t row_bytes = u4 ? hidden / 2 : hidden;
    std::vector<uint8_t> data(rows * row_bytes, 0);
    std::vector<float> scales(hidden);
    std::vector<float> zps(hidden);
    if (u4) {
        CPUTestUtils::scalar_quant_u4_by_channel(src, data.data(), rows, hidden, row_bytes, scales.data(), zps.data());
        CPUTestUtils::scalar_dequant_u4_by_channel(data.data(),
                                                   dst,
                                                   rows,
                                                   hidden,
                                                   row_bytes,
                                                   scales.data(),
                                                   zps.data());
    } else {
        CPUTestUtils::scalar_quant_u8_by_channel(src, data.data(), rows, hidden, row_bytes, scales.data(), zps.data());
        CPUTestUtils::scalar_dequant_u8_by_channel(data.data(),
                                                   dst,
                                                   rows,
                                                   hidden,
                                                   row_bytes,
                                                   scales.data(),
                                                   zps.data());
    }
}

struct ReferenceDims {
    size_t num_heads;
    size_t num_kv_heads;
    size_t num_selection_heads;
    size_t k_head_size;
    size_t v_head_size;
    size_t key_block_size;
    size_t max_selected_blocks;
};

// Independent (host-vector based) re-implementation of the PagedSparseAttention spec for one sequence, used purely as
// a test oracle. `key`/`value` are the FULL logical history of the sequence, [kv, past_len + chunk_len, S], i.e. the
// rows the caches hold once the new tokens are written; `query` covers only the chunk's rows. `sel_indices` /
// `sel_count` are [chunk_len, num_selection_heads, max_selected_blocks] / [chunk_len, num_selection_heads].
std::vector<float> run_sparse_pa_reference(const ReferenceDims& d,
                                           size_t chunk_len,
                                           float scale,
                                           int64_t past_len,
                                           const std::vector<float>& query,
                                           const std::vector<float>& key,
                                           const std::vector<float>& value,
                                           const std::vector<int32_t>& sel_indices,
                                           const std::vector<int32_t>& sel_count) {
    const int64_t history_len = past_len + static_cast<int64_t>(chunk_len);
    auto key_at = [&](size_t head, int64_t pos) {
        return &key[(head * static_cast<size_t>(history_len) + static_cast<size_t>(pos)) * d.k_head_size];
    };
    auto value_at = [&](size_t head, int64_t pos) {
        return &value[(head * static_cast<size_t>(history_len) + static_cast<size_t>(pos)) * d.v_head_size];
    };

    std::vector<float> output(chunk_len * d.num_heads * d.v_head_size, 0.0F);
    const size_t group = d.num_heads / d.num_kv_heads;
    const size_t q_per_sel = d.num_heads / d.num_selection_heads;
    const auto Bk = static_cast<int64_t>(d.key_block_size);
    for (size_t l = 0; l < chunk_len; l++) {
        const int64_t token_pos = past_len + static_cast<int64_t>(l);
        for (size_t h = 0; h < d.num_heads; h++) {
            const size_t kv_head = h / group;
            const size_t sh = h / q_per_sel;
            const float* q = &query[(l * d.num_heads + h) * d.k_head_size];
            std::vector<float> acc(d.v_head_size, 0.0F);
            float running_max = -std::numeric_limits<float>::infinity();
            float running_sum = 0.0F;
            auto consume = [&](int64_t kpos) {
                if (kpos < 0 || kpos >= history_len || kpos > token_pos) {
                    return;
                }
                const float* k = key_at(kv_head, kpos);
                const float* v = value_at(kv_head, kpos);
                float dot = 0.0F;
                for (size_t i = 0; i < d.k_head_size; i++) {
                    dot += q[i] * k[i];
                }
                const float score = dot * scale;
                const float new_max = std::max(running_max, score);
                const float corr = std::exp(running_max - new_max);
                const float p = std::exp(score - new_max);
                running_sum = running_sum * corr + p;
                for (size_t i = 0; i < d.v_head_size; i++) {
                    acc[i] = acc[i] * corr + p * v[i];
                }
                running_max = new_max;
            };

            const size_t row = l * d.num_selection_heads + sh;
            const auto nblocks = static_cast<size_t>(sel_count[row]);
            for (size_t i = 0; i < nblocks; i++) {
                const int64_t block_start = static_cast<int64_t>(sel_indices[row * d.max_selected_blocks + i]) * Bk;
                for (int64_t p = block_start; p < block_start + Bk; p++) {
                    consume(p);
                }
            }

            // sel_indices/sel_count only ever carry *complete* blocks. The consumer derives and attends the
            // incomplete block of the query itself, which is empty when the position ends a block.
            for (int64_t p = ((token_pos + 1) / Bk) * Bk; p <= token_pos; p++) {
                consume(p);
            }

            const float inv_sum = running_sum > 0.0F ? 1.0F / running_sum : 0.0F;
            for (size_t i = 0; i < d.v_head_size; i++) {
                output[(l * d.num_heads + h) * d.v_head_size + i] = acc[i] * inv_sum;
            }
        }
    }
    return output;
}

const char* selection_name(ov::test::SparsePASelection selection) {
    switch (selection) {
    case ov::test::SparsePASelection::kRecent:
        return "Recent";
    case ov::test::SparsePASelection::kRandom:
        return "Random";
    default:
        return "Empty";
    }
}

}  // namespace

namespace ov {
namespace test {

std::string SparsePALayerCPUTest::getTestCaseName(const testing::TestParamInfo<SparsePALayerTestParams>& obj) {
    const auto& [data_precision, shapes, cfg] = obj.param;
    std::ostringstream result;
    result << "Prc=" << data_precision << "_Steps=";
    for (const auto& step : shapes.at(0).second) {
        result << step[0] << "_";
    }
    const auto& first = shapes.at(0).second.at(0);
    result << "B=" << first[1] << "_Hq=" << first[2] << "_Dh=" << first[3] << "_Hkv=" << cfg.num_kv_heads
           << "_Hsel=" << cfg.num_selection_heads << "_Dv=" << cfg.v_head_size << "_Bk=" << cfg.key_block_size
           << "_Page=" << cfg.page_size << "_Kmax=" << cfg.max_selected_blocks
           << "_Sel=" << selection_name(cfg.selection)
           << "_Cache=" << (cfg.cache_precision == ov::element::dynamic ? "plain" : cfg.cache_precision.get_type_name())
           << "_KeyByChannel=" << cfg.key_bychannel << "_Group=" << cfg.group_size;
    return result.str();
}

ov::Shape SparsePALayerCPUTest::cache_shape(bool is_key) const {
    const size_t head = is_key ? m_k_head_size : m_cfg.v_head_size;
    const size_t n_phys = m_num_sequences * m_pages_per_sequence;
    if (!quantized()) {
        return {n_phys, m_cfg.num_kv_heads, m_cfg.page_size, head};
    }
    // Scale and zero point are f32; u4 counts them in half-byte elements.
    const size_t params = m_cfg.cache_precision == ov::element::u4 ? 16 : 8;
    const size_t group = m_cfg.group_size ? m_cfg.group_size : head;
    if (is_key && m_cfg.key_bychannel) {
        return {n_phys, m_cfg.num_kv_heads, m_cfg.page_size + params, head};
    }
    return {n_phys, m_cfg.num_kv_heads, m_cfg.page_size, head + params * (head / group)};
}

void SparsePALayerCPUTest::SetUp() {
    const auto& [data_precision, shapes, cfg] = GetParam();
    targetDevice = ov::test::utils::DEVICE_CPU;
    m_data_precision = data_precision;
    m_cfg = cfg;
    m_steps = shapes.at(0).second;
    ASSERT_FALSE(m_steps.empty());
    m_num_sequences = m_steps[0][1];
    m_num_heads = m_steps[0][2];
    m_k_head_size = m_steps[0][3];
    size_t total_tokens = 0;
    for (const auto& step : m_steps) {
        ASSERT_EQ(step.size(), 4U) << "SparsePA test expects [L, B, H, S] step shapes";
        ASSERT_EQ(step[1], m_num_sequences) << "the number of sequences is fixed across the steps";
        ASSERT_EQ(step[2], m_num_heads);
        ASSERT_EQ(step[3], m_k_head_size);
        total_tokens += step[0];
    }
    m_pages_per_sequence = (total_tokens + m_cfg.page_size - 1) / m_cfg.page_size;
    m_scale = 1.0F / std::sqrt(static_cast<float>(m_k_head_size));

    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    // Every floating-point input follows the data precision, which is also the inference precision.
    configuration.insert(ov::hint::inference_precision(data_precision));
    if (data_precision == ov::element::bf16) {
        rel_threshold = 5e-2F;
        abs_threshold = 5e-2F;
    } else if (data_precision == ov::element::f16) {
        rel_threshold = 5e-3F;
        abs_threshold = 5e-3F;
    }
    // ConvertPagedAttnInputs applies the KV cache precision (u8 by default) to the cache inputs: pin it to the data
    // precision for plain caches, or to the quantized precision, quantization mode and group size under test.
    if (quantized()) {
        configuration[ov::key_cache_precision.name()] = m_cfg.cache_precision;
        configuration[ov::value_cache_precision.name()] = m_cfg.cache_precision;
        configuration[ov::key_cache_group_size.name()] = static_cast<uint64_t>(m_cfg.group_size);
        configuration[ov::value_cache_group_size.name()] = static_cast<uint64_t>(m_cfg.group_size);
        configuration[ov::internal::key_cache_quant_mode.name()] =
            m_cfg.key_bychannel ? ov::internal::CacheQuantMode::BY_CHANNEL : ov::internal::CacheQuantMode::BY_TOKEN;
        configuration[ov::internal::value_cache_quant_mode.name()] = ov::internal::CacheQuantMode::BY_TOKEN;
    } else {
        configuration.insert(ov::hint::kv_cache_precision(data_precision));
    }

    const size_t Hkv = m_cfg.num_kv_heads;
    const size_t Dv = m_cfg.v_head_size;
    const size_t Hs = m_cfg.num_selection_heads;
    const size_t K = m_cfg.max_selected_blocks;
    const size_t n_phys = m_num_sequences * m_pages_per_sequence;
    const auto dyn = ov::Dimension::dynamic();
    const auto dim = [](size_t value) {
        return static_cast<int64_t>(value);
    };

    auto p_query = std::make_shared<ov::op::v0::Parameter>(data_precision,
                                                           ov::PartialShape{dyn, dim(m_num_heads * m_k_head_size)});
    auto p_key =
        std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{dyn, dim(Hkv * m_k_head_size)});
    auto p_value = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{dyn, dim(Hkv * Dv)});
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
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{dyn, dim(Hs), dim(K)});
    auto p_sel_count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{dyn, dim(Hs)});

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

    // Static target shapes of every step, in parameter order.
    std::vector<InputShape> input_shapes;
    for (const auto& parameter : parameters) {
        input_shapes.emplace_back(parameter->get_partial_shape(), std::vector<ov::Shape>{});
    }
    for (const auto& step : m_steps) {
        const size_t tokens = step[0] * m_num_sequences;
        const std::vector<ov::Shape> step_shapes = {{tokens, m_num_heads * m_k_head_size},
                                                    {tokens, Hkv * m_k_head_size},
                                                    {tokens, Hkv * Dv},
                                                    cache_shape(true),
                                                    cache_shape(false),
                                                    {m_num_sequences},
                                                    {m_num_sequences + 1},
                                                    {n_phys},
                                                    {m_num_sequences + 1},
                                                    {tokens, Hs, K},
                                                    {tokens, Hs}};
        for (size_t i = 0; i < step_shapes.size(); i++) {
            input_shapes[i].second.push_back(step_shapes[i]);
        }
    }
    init_input_shapes(input_shapes);

    auto scale = ov::op::v0::Constant::create(data_precision, ov::Shape{}, {m_scale});
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
    auto sparse_pa = std::make_shared<ov::op::internal::PagedSparseAttention>(args, m_cfg.key_block_size);
    if (quantized()) {
        // Head metadata for ConvertPagedAttnInputs, which builds the padded quantized cache shapes from it.
        sparse_pa->get_rt_info()["num_k_heads"] = Hkv;
        sparse_pa->get_rt_info()["k_head_size"] = m_k_head_size;
        sparse_pa->get_rt_info()["num_v_heads"] = Hkv;
        sparse_pa->get_rt_info()["v_head_size"] = Dv;
    }

    function =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sparse_pa)}, parameters);
}

void SparsePALayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    const size_t step = m_step++;
    const size_t L = m_steps.at(step)[0];
    const size_t B = m_num_sequences;
    const size_t Hq = m_num_heads;
    const size_t Dh = m_k_head_size;
    const size_t Hkv = m_cfg.num_kv_heads;
    const size_t Dv = m_cfg.v_head_size;
    const size_t Hs = m_cfg.num_selection_heads;
    const size_t K = m_cfg.max_selected_blocks;
    const size_t Bk = m_cfg.key_block_size;
    const size_t tokens = B * L;
    const size_t n_phys = B * m_pages_per_sequence;
    m_cur_past = m_total_len;
    m_cur_len = L;
    m_total_len += L;

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
    const auto make_i32 = [](const std::vector<int32_t>& values) {
        ov::Tensor t(ov::element::i32, ov::Shape{values.size()});
        std::copy(values.begin(), values.end(), t.data<int32_t>());
        return t;
    };

    // The caches start out empty and are written by the op itself in every step, like a real multi-call sequence.
    if (step == 0) {
        m_key_cache = ov::Tensor(quantized() ? m_cfg.cache_precision : m_data_precision, cache_shape(true));
        m_value_cache = ov::Tensor(quantized() ? m_cfg.cache_precision : m_data_precision, cache_shape(false));
        std::memset(m_key_cache.data(), 0, m_key_cache.get_byte_size());
        std::memset(m_value_cache.data(), 0, m_value_cache.get_byte_size());
    }

    ov::Tensor query(m_data_precision, ov::Shape{tokens, Hq * Dh});
    ov::Tensor key(m_data_precision, ov::Shape{tokens, Hkv * Dh});
    ov::Tensor value(m_data_precision, ov::Shape{tokens, Hkv * Dv});
    {
        std::vector<float> query_data(query.get_size());
        std::vector<float> key_data(key.get_size());
        std::vector<float> value_data(value.get_size());
        for (size_t s = 0; s < B; s++) {
            const uint32_t seed = static_cast<uint32_t>(step) * 101U + static_cast<uint32_t>(s) * 11U;
            std::vector<float> q_slot(L * Hq * Dh);
            std::vector<float> k_slot(L * Hkv * Dh);
            std::vector<float> v_slot(L * Hkv * Dv);
            fill_pseudo_random(q_slot, 1 + seed);
            fill_pseudo_random(k_slot, 2 + seed);
            fill_pseudo_random(v_slot, 3 + seed);
            std::copy(q_slot.begin(), q_slot.end(), query_data.begin() + static_cast<ptrdiff_t>(s * q_slot.size()));
            std::copy(k_slot.begin(), k_slot.end(), key_data.begin() + static_cast<ptrdiff_t>(s * k_slot.size()));
            std::copy(v_slot.begin(), v_slot.end(), value_data.begin() + static_cast<ptrdiff_t>(s * v_slot.size()));
        }
        assign_precision(query, query_data);
        assign_precision(key, key_data);
        assign_precision(value, value_data);
    }

    // Non-identity permutation (simple reversal) across all physical blocks shared by every sequence, so a kernel bug
    // that walked physical blocks in logical order instead of resolving through block_indices would give wrong results.
    std::vector<int32_t> block_indices(n_phys);
    for (size_t i = 0; i < n_phys; i++) {
        block_indices[i] = static_cast<int32_t>(n_phys - 1 - i);
    }

    // Synthetic selection of the complete blocks every token attends to, see make_selection().
    std::vector<int32_t> sel_indices_flat(tokens * Hs * K, -1);
    std::vector<int32_t> sel_count_flat(tokens * Hs, 0);
    uint32_t rng = 12345U + static_cast<uint32_t>(step);
    for (size_t s = 0; s < B; s++) {
        for (size_t l = 0; l < L; l++) {
            const size_t token = s * L + l;
            const size_t num_complete = (m_cur_past + l + 1) / Bk;
            for (size_t sh = 0; sh < Hs; sh++) {
                const size_t row = token * Hs + sh;
                make_selection(m_cfg.selection, K, num_complete, rng, &sel_indices_flat[row * K], sel_count_flat[row]);
            }
        }
    }

    const auto& params = function->get_parameters();
    OPENVINO_ASSERT(targetInputStaticShapes.size() == params.size());
    std::vector<ov::Tensor> tensors(params.size());
    tensors[0] = query;
    tensors[1] = key;
    tensors[2] = value;
    tensors[3] = m_key_cache;
    tensors[4] = m_value_cache;
    tensors[5] = make_i32(std::vector<int32_t>(B, static_cast<int32_t>(m_cur_past)));
    std::vector<int32_t> subsequence_begins(B + 1);
    std::vector<int32_t> block_indices_begins(B + 1);
    for (size_t s = 0; s <= B; s++) {
        subsequence_begins[s] = static_cast<int32_t>(s * L);
        block_indices_begins[s] = static_cast<int32_t>(s * m_pages_per_sequence);
    }
    tensors[6] = make_i32(subsequence_begins);
    tensors[7] = make_i32(block_indices);
    tensors[8] = make_i32(block_indices_begins);
    {
        ov::Tensor t(ov::element::i32, ov::Shape{tokens, Hs, K});
        std::copy(sel_indices_flat.begin(), sel_indices_flat.end(), t.data<int32_t>());
        tensors[9] = t;
    }
    {
        ov::Tensor t(ov::element::i32, ov::Shape{tokens, Hs});
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

    const size_t B = m_num_sequences;
    const size_t L = m_cur_len;
    const size_t past = m_cur_past;
    const size_t total = past + L;
    const size_t Hq = m_num_heads;
    const size_t Dh = m_k_head_size;
    const size_t Hkv = m_cfg.num_kv_heads;
    const size_t Dv = m_cfg.v_head_size;
    const size_t Hs = m_cfg.num_selection_heads;
    const size_t K = m_cfg.max_selected_blocks;
    const size_t page = m_cfg.page_size;
    const auto prec = m_cfg.cache_precision;
    const ReferenceDims dims{Hq, Hkv, Hs, Dh, Dv, m_cfg.key_block_size, K};

    const auto query = to_f32(params[0]);
    const auto key = to_f32(params[1]);
    const auto value = to_f32(params[2]);
    const auto sel_indices = to_i32(params[9]);
    const auto sel_count = to_i32(params[10]);

    if (m_eff_key.empty()) {
        m_eff_key.assign(B, std::vector<std::vector<float>>(Hkv));
        m_eff_value.assign(B, std::vector<std::vector<float>>(Hkv));
    }

    // Oracle for the caches. The effective rows are what the op reads back: plain caches hold the rows as given;
    // quantized by-token rows are quantized once when written; a by-channel key page is requantized as a whole (its
    // dequantized earlier rows plus the new ones) every time new rows land in it, so the rows chain across the steps.
    for (size_t s = 0; s < B; s++) {
        for (size_t kv = 0; kv < Hkv; kv++) {
            auto& eff_key = m_eff_key[s][kv];
            auto& eff_value = m_eff_value[s][kv];
            eff_key.resize(total * Dh);
            eff_value.resize(total * Dv);
            for (size_t l = 0; l < L; l++) {
                const size_t token = s * L + l;
                const float* k_src = &key[(token * Hkv + kv) * Dh];
                const float* v_src = &value[(token * Hkv + kv) * Dv];
                float* k_dst = &eff_key[(past + l) * Dh];
                float* v_dst = &eff_value[(past + l) * Dv];
                if (quantized()) {
                    fake_quant_by_token(prec, m_cfg.group_size, v_src, v_dst, Dv);
                    if (m_cfg.key_bychannel) {
                        std::copy_n(k_src, Dh, k_dst);  // requantized with its page below
                    } else {
                        fake_quant_by_token(prec, m_cfg.group_size, k_src, k_dst, Dh);
                    }
                } else {
                    std::copy_n(k_src, Dh, k_dst);
                    std::copy_n(v_src, Dv, v_dst);
                }
            }
            if (quantized() && m_cfg.key_bychannel) {
                for (size_t page_start = (past / page) * page; page_start < total; page_start += page) {
                    const size_t rows = std::min(page, total - page_start);
                    const std::vector<float> page_rows(
                        eff_key.begin() + static_cast<ptrdiff_t>(page_start * Dh),
                        eff_key.begin() + static_cast<ptrdiff_t>((page_start + rows) * Dh));
                    fake_quant_by_channel(prec, page_rows.data(), &eff_key[page_start * Dh], rows, Dh);
                }
            }
        }
    }

    ov::Tensor output(ov::element::f32, ov::Shape{B * L, Hq * Dv});
    auto* output_data = output.data<float>();
    for (size_t s = 0; s < B; s++) {
        // Full logical history of the sequence as [kv, total, S].
        std::vector<float> history_key;
        std::vector<float> history_value;
        for (size_t kv = 0; kv < Hkv; kv++) {
            history_key.insert(history_key.end(), m_eff_key[s][kv].begin(), m_eff_key[s][kv].end());
            history_value.insert(history_value.end(), m_eff_value[s][kv].begin(), m_eff_value[s][kv].end());
        }
        const auto first = static_cast<ptrdiff_t>(s * L);
        const auto last = first + static_cast<ptrdiff_t>(L);
        const std::vector<float> query_s(query.begin() + first * static_cast<ptrdiff_t>(Hq * Dh),
                                         query.begin() + last * static_cast<ptrdiff_t>(Hq * Dh));
        const std::vector<int32_t> sel_indices_s(sel_indices.begin() + first * static_cast<ptrdiff_t>(Hs * K),
                                                 sel_indices.begin() + last * static_cast<ptrdiff_t>(Hs * K));
        const std::vector<int32_t> sel_count_s(sel_count.begin() + first * static_cast<ptrdiff_t>(Hs),
                                               sel_count.begin() + last * static_cast<ptrdiff_t>(Hs));
        const auto out_s = run_sparse_pa_reference(dims,
                                                   L,
                                                   m_scale,
                                                   static_cast<int64_t>(past),
                                                   query_s,
                                                   history_key,
                                                   history_value,
                                                   sel_indices_s,
                                                   sel_count_s);
        std::copy(out_s.begin(), out_s.end(), output_data + s * L * Hq * Dv);
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
