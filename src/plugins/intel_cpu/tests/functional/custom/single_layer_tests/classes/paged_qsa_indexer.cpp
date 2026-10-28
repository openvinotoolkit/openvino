// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

// Fixed small configuration exercised by this test: 2 independent sequences, each with its own
// past_len (kPastLens[s]) and its own run of kSeqBlocks physical cache blocks, addressed through
// a non-identity BLOCK_INDICES permutation (see generate_inputs) so a kernel bug that walked
// physical blocks in logical/sequential order instead of resolving through BLOCK_INDICES would be
// caught. kL tokens is not a multiple of compress_ratio, so every instantiation naturally
// exercises the causal-tail-marker path on its final tokens and the empty-selection path on its
// first token -- mirroring the dense QSAIndexer test (classes/qsa_indexer.cpp), whose per-token
// math this test's oracle reuses verbatim: paged and dense kernels share identical per-token/
// per-block math and process sequences independently, differing only in cache addressing.
constexpr size_t kL = 7;
constexpr size_t kDidx = 16;
constexpr size_t kRotaryDim = 8;
constexpr size_t kR = 2;
constexpr size_t kBlockTopk = 2;
constexpr std::array<int32_t, 2> kPastLens = {0, 2};
constexpr size_t kSCap = kL + static_cast<size_t>(kPastLens.back());
constexpr size_t kSummaryLen = kSCap / kR;
constexpr float kEps = 1e-6F;
constexpr float kScale = 0.25F;

// Paged-cache layout: page_size must be a multiple of compress_ratio. kSeqBlocks physical blocks
// per sequence give a raw-k token capacity of kSeqBlocks*kPageSize == 16 tokens and a summary/
// position-cache slot capacity of kSeqBlocks*(kPageSize/kR) == 8 blocks, both comfortably above
// this test's worst-case single-sequence requirement (kSCap == 9 tokens, kSummaryLen == 4 blocks).
constexpr size_t kPageSize = 4;
constexpr size_t kSeqBlocks = 4;
constexpr size_t kSlotsPerPage = kPageSize / kR;

void fill_pseudo_random(std::vector<float>& v, uint32_t seed_base) {
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t x = seed_base * 2654435761U + static_cast<uint32_t>(i) * 40503U + 1U;
        x ^= x >> 13;
        x *= 0x85EBCA6BU;
        x ^= x >> 16;
        v[i] = (static_cast<float>(x % 20000U) / 10000.0F) - 1.0F;
    }
}

float rms_inv_scale(const float* x, size_t n, float eps) {
    float sumsq = 0.0F;
    for (size_t i = 0; i < n; i++) {
        sumsq += x[i] * x[i];
    }
    return 1.0F / std::sqrt(sumsq / static_cast<float>(n) + eps);
}

void rmsnorm_partial_rope(const float* src,
                          float* dst,
                          const float* weight,
                          size_t dim,
                          size_t rotary_dim,
                          float eps,
                          const float* cos_sin_row) {
    const float inv_rms = rms_inv_scale(src, dim, eps);
    for (size_t d = 0; d < dim; d++) {
        dst[d] = src[d] * inv_rms * weight[d];
    }
    const size_t half = rotary_dim / 2;
    for (size_t j = 0; j < half; j++) {
        const float cos_lo = cos_sin_row[2 * j];
        const float sin_lo = cos_sin_row[2 * j + 1];
        const float cos_hi = cos_sin_row[2 * (j + half)];
        const float sin_hi = cos_sin_row[2 * (j + half) + 1];
        const float x0 = dst[j];
        const float x1 = dst[j + half];
        dst[j] = x0 * cos_lo - x1 * sin_lo;
        dst[j + half] = x1 * cos_hi + x0 * sin_hi;
    }
}

// Independent (host-vector based) re-implementation of the paged-qsa-indexer.rst per-sequence
// spec, used purely as a test oracle -- NOT a copy of the CPU node's PlainTensor-based paged
// addressing. Per-sequence math is bit-for-bit identical to the dense QSAIndexer oracle
// (classes/qsa_indexer.cpp's run_qsa_indexer_reference): the paged kernel's logical_pos ==
// past_len + local token index, exactly dense's abs_pos, and sequences never interact.
struct PagedQSAIndexerOracleOutput {
    std::vector<int64_t> sel_indices;
    std::vector<int64_t> sel_count;
};

PagedQSAIndexerOracleOutput run_paged_qsa_indexer_reference(size_t h_iq,
                                                            size_t h_ik,
                                                            size_t h_sel,
                                                            const std::vector<float>& q_idx,
                                                            const std::vector<float>& k_idx,
                                                            const std::vector<float>& rope_cos,
                                                            const std::vector<float>& rope_sin,
                                                            const std::vector<float>& q_norm_weight,
                                                            const std::vector<float>& k_norm_weight,
                                                            int64_t past_len) {
    const size_t group = h_iq / h_ik;
    std::vector<size_t> kv_of_head(h_iq), sel_of_head(h_iq);
    for (size_t h = 0; h < h_iq; h++) {
        kv_of_head[h] = h / group;
        sel_of_head[h] = (kv_of_head[h] * h_sel) / h_ik;
    }

    std::vector<float> raw_k_cache(h_ik * kSCap * kDidx, 0.0F);
    std::vector<float> summary_cache(h_ik * kSummaryLen * kDidx, 0.0F);

    const auto interleave = [&](size_t local_idx, std::array<float, 2 * kRotaryDim>& buf) {
        for (size_t j = 0; j < kRotaryDim; j++) {
            buf[2 * j] = rope_cos[local_idx * kRotaryDim + j];
            buf[2 * j + 1] = rope_sin[local_idx * kRotaryDim + j];
        }
    };

    PagedQSAIndexerOracleOutput out;
    out.sel_indices.assign(h_sel * kL * kBlockTopk, -1);
    out.sel_count.assign(h_sel * kL, 0);

    for (size_t l = 0; l < kL; l++) {
        const int64_t abs_pos = past_len + static_cast<int64_t>(l);

        std::vector<float> q_normed(h_iq * kDidx);
        std::array<float, 2 * kRotaryDim> cos_sin_buf{};
        interleave(l, cos_sin_buf);
        const float* cos_sin_row = cos_sin_buf.data();
        for (size_t h = 0; h < h_iq; h++) {
            const float* src = &q_idx[(l * h_iq + h) * kDidx];
            rmsnorm_partial_rope(src, &q_normed[h * kDidx], q_norm_weight.data(), kDidx, kRotaryDim, kEps, cos_sin_row);
        }

        for (size_t h = 0; h < h_ik; h++) {
            const float* src = &k_idx[(l * h_ik + h) * kDidx];
            std::copy_n(src, kDidx, &raw_k_cache[(h * kSCap + static_cast<size_t>(abs_pos)) * kDidx]);
        }

        const int64_t n_complete = (abs_pos + 1) / static_cast<int64_t>(kR);
        if ((abs_pos + 1) % static_cast<int64_t>(kR) == 0) {
            const int64_t block_idx = n_complete - 1;
            const int64_t block_start = block_idx * static_cast<int64_t>(kR);
            std::array<float, 2 * kRotaryDim> block_cos_sin_buf{};
            interleave(static_cast<size_t>(block_start - past_len), block_cos_sin_buf);
            const float* block_cos_sin = block_cos_sin_buf.data();
            for (size_t h = 0; h < h_ik; h++) {
                std::vector<float> pooled(kDidx, 0.0F);
                for (int64_t p = block_start; p < block_start + static_cast<int64_t>(kR); p++) {
                    const float* raw = &raw_k_cache[(h * kSCap + static_cast<size_t>(p)) * kDidx];
                    for (size_t d = 0; d < kDidx; d++) {
                        pooled[d] += raw[d];
                    }
                }
                for (float& v : pooled) {
                    v /= static_cast<float>(kR);
                }
                std::vector<float> summary(kDidx);
                rmsnorm_partial_rope(pooled.data(),
                                     summary.data(),
                                     k_norm_weight.data(),
                                     kDidx,
                                     kRotaryDim,
                                     kEps,
                                     block_cos_sin);
                std::copy(summary.begin(),
                          summary.end(),
                          &summary_cache[(h * kSummaryLen + static_cast<size_t>(block_idx)) * kDidx]);
            }
        }

        for (size_t sh = 0; sh < h_sel; sh++) {
            size_t k = 0;
            std::vector<int32_t> chosen;
            if (n_complete > 0) {
                std::vector<std::pair<float, int32_t>> scores(static_cast<size_t>(n_complete));
                for (int64_t bidx = 0; bidx < n_complete; bidx++) {
                    float score = 0.0F;
                    for (size_t h = 0; h < h_iq; h++) {
                        if (sel_of_head[h] != sh) {
                            continue;
                        }
                        const float* summary =
                            &summary_cache[(kv_of_head[h] * kSummaryLen + static_cast<size_t>(bidx)) * kDidx];
                        float dot = 0.0F;
                        for (size_t d = 0; d < kDidx; d++) {
                            dot += q_normed[h * kDidx + d] * summary[d];
                        }
                        score += std::max(0.0F, dot);
                    }
                    scores[static_cast<size_t>(bidx)] = {score * kScale, static_cast<int32_t>(bidx)};
                }
                k = std::min(kBlockTopk, static_cast<size_t>(n_complete));
                std::partial_sort(scores.begin(),
                                  scores.begin() + static_cast<ptrdiff_t>(k),
                                  scores.end(),
                                  [](const auto& a, const auto& b) {
                                      if (a.first != b.first) {
                                          return a.first > b.first;
                                      }
                                      return a.second < b.second;
                                  });
                chosen.resize(k);
                for (size_t i = 0; i < k; i++) {
                    chosen[i] = scores[i].second;
                }
                std::sort(chosen.begin(), chosen.end());
            }

            auto* out_row = &out.sel_indices[(sh * kL + l) * kBlockTopk];
            for (size_t i = 0; i < k; i++) {
                out_row[i] = chosen[i];
            }
            out.sel_count[sh * kL + l] = static_cast<int64_t>(k);
        }
    }
    return out;
}

}  // namespace

namespace ov {
namespace test {

std::string PagedQSAIndexerLayerCPUTest::getTestCaseName(
    const testing::TestParamInfo<PagedQSAIndexerLayerTestParams>& obj) {
    const auto& [device, data_precision, h_iq, h_ik, h_sel, num_sequences] = obj.param;
    return "Target=" + device + "_DataPrecision=" + data_precision.get_type_name() + "_Hiq=" + std::to_string(h_iq) +
           "_Hik=" + std::to_string(h_ik) + "_Hsel=" + std::to_string(h_sel) +
           "_NumSeq=" + std::to_string(num_sequences);
}

void PagedQSAIndexerLayerCPUTest::SetUp() {
    const auto& [device, data_precision, h_iq, h_ik, h_sel, num_sequences] = GetParam();
    targetDevice = device;
    m_data_precision = data_precision;
    m_h_iq = h_iq;
    m_h_ik = h_ik;
    m_h_sel = h_sel;
    m_num_sequences = num_sequences;
    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    if (data_precision == ov::element::bf16) {
        rel_threshold = 5e-2F;
        abs_threshold = 5e-2F;
        configuration.insert(ov::hint::inference_precision(ov::element::bf16));
    } else if (data_precision == ov::element::f16) {
        rel_threshold = 5e-3F;
        abs_threshold = 5e-3F;
        configuration.insert(ov::hint::inference_precision(ov::element::f16));
    }

    const size_t tokens_total = m_num_sequences * kL;
    const size_t n_phys = m_num_sequences * kSeqBlocks;

    auto p_q_idx = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{tokens_total, m_h_iq * kDidx});
    auto p_k_idx = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{tokens_total, m_h_ik * kDidx});
    auto p_rope_cos = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{tokens_total, kRotaryDim});
    auto p_rope_sin = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{tokens_total, kRotaryDim});
    auto p_q_norm_weight = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{kDidx});
    auto p_k_norm_weight = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{kDidx});
    auto p_past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences});
    auto p_subsequence_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    auto p_block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{n_phys});
    auto p_block_indices_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    auto p_raw_k_cache =
        std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{n_phys, m_h_ik, kPageSize, kDidx});
    auto p_indexer_rope_cache =
        std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{n_phys, kSlotsPerPage, 2 * kRotaryDim});
    auto p_summary_cache =
        std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{n_phys, m_h_ik, kSlotsPerPage, kDidx});

    const ov::ParameterVector params_in{p_q_idx,
                                        p_k_idx,
                                        p_rope_cos,
                                        p_rope_sin,
                                        p_q_norm_weight,
                                        p_k_norm_weight,
                                        p_past_lens,
                                        p_subsequence_begins,
                                        p_block_indices,
                                        p_block_indices_begins,
                                        p_raw_k_cache,
                                        p_indexer_rope_cache,
                                        p_summary_cache};
    std::vector<ov::Shape> input_shapes;
    for (const auto& p : params_in) {
        input_shapes.push_back(p->get_shape());
    }
    init_input_shapes(static_shapes_to_test_representation(input_shapes));

    ov::op::internal::PagedQSAIndexer::Config config;
    config.key_block_size = kR;
    config.block_topk = kBlockTopk;
    config.num_selection_heads = m_h_sel;
    config.indexer_rotary_dim = kRotaryDim;
    config.indexer_head_dim = kDidx;
    config.epsilon = kEps;
    config.scale = kScale;

    ov::OutputVector args(params_in.begin(), params_in.end());
    auto paged_qsa_indexer = std::make_shared<ov::op::internal::PagedQSAIndexer>(args, config);

    function = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(paged_qsa_indexer->output(0)),
                         std::make_shared<ov::op::v0::Result>(paged_qsa_indexer->output(1))},
        params_in);
}

void PagedQSAIndexerLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
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
    // Each sequence gets its own pseudo-random seed (seed_base + s * 97) so sequence 0 and
    // sequence 1 data are distinguishable -- a kernel bug that mixed up the sequence index (or
    // its physical cache blocks) would otherwise go undetected since all sequences would look
    // identical.
    const auto make_seq_data = [this](size_t h, uint32_t seed_base) {
        ov::Tensor t(m_data_precision, ov::Shape{m_num_sequences * kL, h * kDidx});
        const size_t per_seq = kL * h * kDidx;
        std::vector<float> data(t.get_size());
        for (size_t s = 0; s < m_num_sequences; s++) {
            std::vector<float> seq_data(per_seq);
            fill_pseudo_random(seq_data, seed_base + static_cast<uint32_t>(s) * 97U);
            std::copy(seq_data.begin(), seq_data.end(), data.begin() + static_cast<ptrdiff_t>(s * per_seq));
        }
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
        return t;
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
    const auto make_norm_weight = [this, &assign_precision](uint32_t seed) {
        std::vector<float> data(kDidx);
        fill_pseudo_random(data, seed);
        for (auto& v : data) {
            v = 1.0F + 0.1F * v;
        }
        ov::Tensor t(m_data_precision, ov::Shape{kDidx});
        assign_precision(t, data);
        return t;
    };

    const size_t n_phys = m_num_sequences * kSeqBlocks;

    const auto& params = function->get_parameters();
    std::vector<ov::Tensor> tensors(params.size());
    using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
    tensors[InputIdx::INDEX_QUERY] = make_seq_data(m_h_iq, 1);
    tensors[InputIdx::RAW_INDEX_KEYS] = make_seq_data(m_h_ik, 2);
    {
        // theta is a function of each token's own absolute position (kPastLens[s] + local index),
        // matching the per-token rope_cos/rope_sin ABI -- no shared position-indexed table.
        // Rows are indexer_rotary_dim wide, one coefficient per rotary channel. The two halves are
        // deliberately NOT duplicates, so a kernel assuming cos[j] == cos[j + rotary_dim/2] fails.
        ov::Tensor t_cos(m_data_precision, ov::Shape{m_num_sequences * kL, kRotaryDim});
        ov::Tensor t_sin(m_data_precision, ov::Shape{m_num_sequences * kL, kRotaryDim});
        std::vector<float> cos_data(m_num_sequences * kL * kRotaryDim);
        std::vector<float> sin_data(m_num_sequences * kL * kRotaryDim);
        for (size_t s = 0; s < m_num_sequences; s++) {
            for (size_t l = 0; l < kL; l++) {
                const size_t abs_pos = static_cast<size_t>(kPastLens[s]) + l;
                const size_t row = s * kL + l;
                for (size_t j = 0; j < kRotaryDim; j++) {
                    const float theta = static_cast<float>(abs_pos) * 0.1F + static_cast<float>(j) * 0.05F;
                    cos_data[row * kRotaryDim + j] = std::cos(theta);
                    sin_data[row * kRotaryDim + j] = std::sin(theta);
                }
            }
        }
        assign_precision(t_cos, cos_data);
        assign_precision(t_sin, sin_data);
        tensors[InputIdx::ROPE_COS] = t_cos;
        tensors[InputIdx::ROPE_SIN] = t_sin;
    }
    tensors[InputIdx::Q_NORM_WEIGHT] = make_norm_weight(6);
    tensors[InputIdx::K_NORM_WEIGHT] = make_norm_weight(7);
    {
        // Non-identity permutation (simple reversal) across all physical blocks shared by every
        // sequence, so a kernel bug that walked physical blocks in logical/sequential order
        // instead of resolving through block_indices would produce numerically wrong results.
        std::vector<int32_t> block_indices(n_phys);
        for (size_t i = 0; i < n_phys; i++) {
            block_indices[i] = static_cast<int32_t>(n_phys - 1 - i);
        }
        tensors[InputIdx::BLOCK_INDICES] = make_i32(block_indices);
    }
    {
        std::vector<int32_t> block_indices_begins(m_num_sequences + 1);
        for (size_t s = 0; s <= m_num_sequences; s++) {
            block_indices_begins[s] = static_cast<int32_t>(s * kSeqBlocks);
        }
        tensors[InputIdx::BLOCK_INDICES_BEGINS] = make_i32(block_indices_begins);
    }
    {
        std::vector<int32_t> subsequence_begins(m_num_sequences + 1);
        for (size_t s = 0; s <= m_num_sequences; s++) {
            subsequence_begins[s] = static_cast<int32_t>(s * kL);
        }
        tensors[InputIdx::SUBSEQUENCE_BEGINS] = make_i32(subsequence_begins);
    }
    {
        std::vector<int32_t> past_lens_values(m_num_sequences);
        for (size_t s = 0; s < m_num_sequences; s++) {
            past_lens_values[s] = kPastLens[s];
        }
        tensors[InputIdx::PAST_LENS] = make_i32(past_lens_values);
    }
    tensors[InputIdx::INDEX_KEY_CACHE] = make_zeros(ov::Shape{n_phys, m_h_ik, kPageSize, kDidx});
    tensors[InputIdx::INDEX_BLOCK_ROPE_CACHE] = make_zeros(ov::Shape{n_phys, kSlotsPerPage, 2 * kRotaryDim});
    tensors[InputIdx::BLOCK_INDEX_KEYS_CACHE] = make_zeros(ov::Shape{n_phys, m_h_ik, kSlotsPerPage, kDidx});

    for (size_t i = 0; i < params.size(); i++) {
        host_inputs[params[i]] = tensors[i];
        inputs[params[i]] = tensors[i];
    }
}

std::vector<ov::Tensor> PagedQSAIndexerLayerCPUTest::calculate_refs() {
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

    using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
    auto q_idx = to_f32(params[InputIdx::INDEX_QUERY]);
    auto k_idx = to_f32(params[InputIdx::RAW_INDEX_KEYS]);
    auto rope_cos = to_f32(params[InputIdx::ROPE_COS]);
    auto rope_sin = to_f32(params[InputIdx::ROPE_SIN]);
    auto q_norm_weight = to_f32(params[InputIdx::Q_NORM_WEIGHT]);
    auto k_norm_weight = to_f32(params[InputIdx::K_NORM_WEIGHT]);
    const auto* past_lens_ptr = host_inputs.at(params[InputIdx::PAST_LENS]).data<const int32_t>();

    const size_t tokens_total = m_num_sequences * kL;
    ov::Tensor sel_indices(ov::element::i32, ov::Shape{tokens_total, m_h_sel, kBlockTopk});
    ov::Tensor sel_count(ov::element::i32, ov::Shape{tokens_total, m_h_sel});
    auto* sel_indices_data = sel_indices.data<int32_t>();
    auto* sel_count_data = sel_count.data<int32_t>();

    const size_t q_idx_stride = kL * m_h_iq * kDidx;
    const size_t k_idx_stride = kL * m_h_ik * kDidx;
    const size_t rope_stride = kL * kRotaryDim;
    for (size_t s = 0; s < m_num_sequences; s++) {
        std::vector<float> q_idx_s(q_idx.begin() + static_cast<ptrdiff_t>(s * q_idx_stride),
                                   q_idx.begin() + static_cast<ptrdiff_t>((s + 1) * q_idx_stride));
        std::vector<float> k_idx_s(k_idx.begin() + static_cast<ptrdiff_t>(s * k_idx_stride),
                                   k_idx.begin() + static_cast<ptrdiff_t>((s + 1) * k_idx_stride));
        std::vector<float> rope_cos_s(rope_cos.begin() + static_cast<ptrdiff_t>(s * rope_stride),
                                      rope_cos.begin() + static_cast<ptrdiff_t>((s + 1) * rope_stride));
        std::vector<float> rope_sin_s(rope_sin.begin() + static_cast<ptrdiff_t>(s * rope_stride),
                                      rope_sin.begin() + static_cast<ptrdiff_t>((s + 1) * rope_stride));
        const int64_t past_len = past_lens_ptr[s];

        auto oracle = run_paged_qsa_indexer_reference(m_h_iq,
                                                      m_h_ik,
                                                      m_h_sel,
                                                      q_idx_s,
                                                      k_idx_s,
                                                      rope_cos_s,
                                                      rope_sin_s,
                                                      q_norm_weight,
                                                      k_norm_weight,
                                                      past_len);

        // Oracle output is [h_sel, kL, kBlockTopk]/[h_sel, kL] (per-sequence, sh-major); the op's
        // real outputs are token-major ([tokens_total, h_sel, kBlockTopk]/[tokens_total, h_sel]),
        // so transpose while scattering into the global, token-offset tensor.
        for (size_t l = 0; l < kL; l++) {
            const size_t token = s * kL + l;
            for (size_t sh = 0; sh < m_h_sel; sh++) {
                for (size_t i = 0; i < kBlockTopk; i++) {
                    sel_indices_data[(token * m_h_sel + sh) * kBlockTopk + i] =
                        static_cast<int32_t>(oracle.sel_indices[(sh * kL + l) * kBlockTopk + i]);
                }
                sel_count_data[token * m_h_sel + sh] = static_cast<int32_t>(oracle.sel_count[sh * kL + l]);
            }
        }
    }
    return {sel_indices, sel_count};
}

void PagedQSAIndexerLayerCPUTest::compare(const std::vector<ov::Tensor>& expected,
                                          const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

TEST_P(PagedQSAIndexerLayerCPUTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov