// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"

namespace ov {
namespace test {

std::vector<float> read_tensor_to_f32(const ov::Tensor& t) {
    std::vector<float> out(t.get_size());
    if (t.get_element_type() == ov::element::bf16) {
        const auto* src = t.data<ov::bfloat16>();
        std::copy(src, src + out.size(), out.begin());
    } else if (t.get_element_type() == ov::element::f16) {
        const auto* src = t.data<ov::float16>();
        std::copy(src, src + out.size(), out.begin());
    } else {
        const auto* src = t.data<float>();
        std::copy(src, src + out.size(), out.begin());
    }
    return out;
}

void write_f32_to_tensor(ov::Tensor& t, const std::vector<float>& data) {
    if (t.get_element_type() == ov::element::bf16) {
        auto* dst = t.data<ov::bfloat16>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::bfloat16(data[i]);
        }
    } else if (t.get_element_type() == ov::element::f16) {
        auto* dst = t.data<ov::float16>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
    } else {
        std::copy(data.begin(), data.end(), t.data<float>());
    }
}

ov::Tensor make_zero_tensor(const ov::element::Type& precision, const ov::Shape& shape) {
    ov::Tensor t(precision, shape);
    if (t.get_byte_size() != 0) {
        std::memset(t.data(), 0, t.get_byte_size());
    }
    return t;
}

namespace {

float rms_inv_scale(const float* x, size_t n, float eps) {
    float sumsq = 0.0F;
    for (size_t i = 0; i < n; i++) {
        sumsq += x[i] * x[i];
    }
    return 1.0F / std::sqrt(sumsq / static_cast<float>(n) + eps);
}

// Independent re-implementation of new_spec/qsa-prepare-17.rst's rotate-half RMSNorm+RoPE, used
// purely as a test oracle. `cos_row`/`sin_row` each hold `rotary_dim` values.
void rmsnorm_partial_rope(const float* src,
                          float* dst,
                          const float* weight,
                          size_t dim,
                          size_t rotary_dim,
                          float eps,
                          const float* cos_row,
                          const float* sin_row) {
    const float inv_rms = rms_inv_scale(src, dim, eps);
    for (size_t d = 0; d < dim; d++) {
        dst[d] = src[d] * inv_rms * weight[d];
    }
    const size_t half = rotary_dim / 2;
    for (size_t j = 0; j < half; j++) {
        const float x0 = dst[j];
        const float x1 = dst[j + half];
        dst[j] = x0 * cos_row[j] - x1 * sin_row[j];
        dst[j + half] = x1 * cos_row[j + half] + x0 * sin_row[j + half];
    }
}

// Independent (host-vector based) oracle for one QSAIndexer call, re-implementing
// new_spec/qsa-prepare-17.rst's padding-aware prepare+selection pseudocode directly -- NOT a copy
// of the CPU node's PlainTensor-based indexing/control flow. `summary_cache`/`past_raw_k`/
// `past_rope` are caller-owned state threaded in by reference across sequential calls, mirroring
// the production op's state (fed back from the op's state outputs).
}  // namespace

QSAIndexerOracle::Output QSAIndexerOracle::run_call(size_t batch,
                                                    int64_t length,
                                                    const std::vector<int32_t>& key_length,
                                                    const std::vector<float>& q_idx,
                                                    const std::vector<float>& k_idx,
                                                    const std::vector<float>& rope_cos,
                                                    const std::vector<float>& rope_sin,
                                                    const std::vector<float>& q_norm_weight,
                                                    const std::vector<float>& k_norm_weight) {
    const size_t group = h_iq / h_ik;
    std::vector<size_t> kv_of_head(h_iq), sel_of_head(h_iq);
    for (size_t h = 0; h < h_iq; h++) {
        kv_of_head[h] = h / group;
        sel_of_head[h] = (kv_of_head[h] * h_sel) / h_ik;
    }

    Output out;
    out.sel_indices.assign(batch * h_sel * static_cast<size_t>(length) * block_topk, -1);
    out.sel_count.assign(batch * h_sel * static_cast<size_t>(length), 0);

    const size_t L = static_cast<size_t>(length);

    for (size_t b = 0; b < batch; b++) {
        const int64_t key_len = key_length[b];
        const int64_t past_n = std::max<int64_t>(0, key_len - length);
        const int64_t current_n = key_len - past_n;
        const int64_t num_padding_n = length - current_n;
        const int64_t c_n = past_n % r;
        const int64_t num_cached_blocks_n = past_n / r;

        std::vector<float> q_normed(h_iq * d_idx);
        std::vector<float> rope_cos_buf(rotary_dim);
        std::vector<float> rope_sin_buf(rotary_dim);
        std::vector<float> block_ref_cos_sin(2 * rotary_dim, 0.0F);
        std::vector<float> pooled(d_idx);
        std::vector<float> summary_buf(d_idx);

        std::vector<float> pending_rows(static_cast<size_t>(r) * h_ik * d_idx, 0.0F);
        auto pending_ptr = [&](int64_t pos_in_block, size_t h) {
            return pending_rows.data() + (static_cast<size_t>(pos_in_block) * h_ik + h) * d_idx;
        };
        auto past_raw_k_ptr = [&](size_t h, int64_t p) {
            return past_raw_k.data() + ((b * h_ik + h) * static_cast<size_t>(r - 1) + static_cast<size_t>(p)) * d_idx;
        };
        auto summary_ptr = [&](size_t h, int64_t block) {
            return summary_cache.data() + ((b * h_ik + h) * summary_cap + static_cast<size_t>(block)) * d_idx;
        };

        for (int64_t p = 0; p < c_n; p++) {
            for (size_t h = 0; h < h_ik; h++) {
                std::copy_n(past_raw_k_ptr(h, p), d_idx, pending_ptr(p, h));
            }
        }
        if (c_n > 0) {
            std::copy_n(past_rope.data() + b * 2 * rotary_dim, 2 * rotary_dim, block_ref_cos_sin.data());
        }

        int64_t running_block_idx = num_cached_blocks_n;

        for (int64_t l = num_padding_n; l < length; l++) {
            const int64_t v = l - num_padding_n;
            const int64_t pending_pos = c_n + v;
            const int64_t pos_in_block = pending_pos % r;
            const size_t lu = static_cast<size_t>(l);

            std::copy_n(&rope_cos[(b * L + lu) * rotary_dim], rotary_dim, rope_cos_buf.data());
            std::copy_n(&rope_sin[(b * L + lu) * rotary_dim], rotary_dim, rope_sin_buf.data());
            for (size_t h = 0; h < h_iq; h++) {
                const float* src = &q_idx[((b * L + lu) * h_iq + h) * d_idx];
                rmsnorm_partial_rope(src,
                                     &q_normed[h * d_idx],
                                     q_norm_weight.data(),
                                     d_idx,
                                     rotary_dim,
                                     eps,
                                     rope_cos_buf.data(),
                                     rope_sin_buf.data());
            }

            if (pos_in_block == 0) {
                std::copy(rope_cos_buf.begin(), rope_cos_buf.end(), block_ref_cos_sin.begin());
                std::copy(rope_sin_buf.begin(),
                          rope_sin_buf.end(),
                          block_ref_cos_sin.begin() + static_cast<ptrdiff_t>(rotary_dim));
            }
            for (size_t h = 0; h < h_ik; h++) {
                std::copy_n(&k_idx[((b * L + lu) * h_ik + h) * d_idx], d_idx, pending_ptr(pos_in_block, h));
            }

            if (pos_in_block + 1 == r) {
                for (size_t h = 0; h < h_ik; h++) {
                    std::fill(pooled.begin(), pooled.end(), 0.0F);
                    for (int64_t p = 0; p < r; p++) {
                        const float* src = pending_ptr(p, h);
                        for (size_t d = 0; d < d_idx; d++) {
                            pooled[d] += src[d];
                        }
                    }
                    const float inv_r = 1.0F / static_cast<float>(r);
                    for (float& val : pooled) {
                        val *= inv_r;
                    }
                    rmsnorm_partial_rope(pooled.data(),
                                         summary_buf.data(),
                                         k_norm_weight.data(),
                                         d_idx,
                                         rotary_dim,
                                         eps,
                                         block_ref_cos_sin.data(),
                                         block_ref_cos_sin.data() + rotary_dim);
                    std::copy(summary_buf.begin(), summary_buf.end(), summary_ptr(h, running_block_idx));
                }
                running_block_idx += 1;
            }

            const int64_t n_complete = running_block_idx;
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
                            const float* summary = summary_ptr(kv_of_head[h], bidx);
                            float dot = 0.0F;
                            for (size_t d = 0; d < d_idx; d++) {
                                dot += q_normed[h * d_idx + d] * summary[d];
                            }
                            score += std::max(0.0F, dot);
                        }
                        scores[static_cast<size_t>(bidx)] = {score * scale, static_cast<int32_t>(bidx)};
                    }
                    k = std::min(block_topk, static_cast<size_t>(n_complete));
                    std::partial_sort(scores.begin(),
                                      scores.begin() + static_cast<ptrdiff_t>(k),
                                      scores.end(),
                                      [](const auto& a, const auto& bb) {
                                          if (a.first != bb.first) {
                                              return a.first > bb.first;
                                          }
                                          return a.second < bb.second;
                                      });
                    chosen.resize(k);
                    for (size_t i = 0; i < k; i++) {
                        chosen[i] = scores[i].second;
                    }
                    std::sort(chosen.begin(), chosen.end());
                }
                auto* out_row = &out.sel_indices[((b * h_sel + sh) * L + lu) * block_topk];
                for (size_t i = 0; i < k; i++) {
                    out_row[i] = chosen[i];
                }
                out.sel_count[(b * h_sel + sh) * L + lu] = static_cast<int32_t>(k);
            }
        }

        const int64_t rest = (c_n + current_n) - (running_block_idx - num_cached_blocks_n) * r;
        for (int64_t p = 0; p < r - 1; p++) {
            for (size_t h = 0; h < h_ik; h++) {
                const float* src = p < rest ? pending_ptr(p, h) : nullptr;
                if (src != nullptr) {
                    std::copy_n(src, d_idx, past_raw_k_ptr(h, p));
                } else {
                    std::fill_n(past_raw_k_ptr(h, p), d_idx, 0.0F);
                }
            }
        }
        if (rest == 0) {
            std::fill(block_ref_cos_sin.begin(), block_ref_cos_sin.end(), 0.0F);
        }
        std::copy(block_ref_cos_sin.begin(), block_ref_cos_sin.end(), past_rope.data() + b * 2 * rotary_dim);
    }
    return out;
}

std::string QSAIndexerLayerCPUTest::getTestCaseName(const testing::TestParamInfo<QSAIndexerLayerTestParams>& obj) {
    const auto& [device, data_precision, h_iq, h_ik, h_sel, batch, scenario] = obj.param;
    std::string scenario_name;
    switch (scenario) {
    case QSAIndexerScenario::kDensePrefill:
        scenario_name = "DensePrefill";
        break;
    case QSAIndexerScenario::kWorkedExample:
        scenario_name = "WorkedExample";
        break;
    case QSAIndexerScenario::kFullyPaddedRow:
        scenario_name = "FullyPaddedRow";
        break;
    case QSAIndexerScenario::kRaggedCalls:
        scenario_name = "RaggedCalls";
        break;
    }
    return "Target=" + device + "_DataPrecision=" + data_precision.get_type_name() + "_Hiq=" + std::to_string(h_iq) +
           "_Hik=" + std::to_string(h_ik) + "_Hsel=" + std::to_string(h_sel) + "_Batch=" + std::to_string(batch) +
           "_Scenario=" + scenario_name;
}

void QSAIndexerLayerCPUTest::SetUp() {
    const auto& [device, data_precision, h_iq, h_ik, h_sel, batch, scenario] = GetParam();
    targetDevice = device;
    m_data_precision = data_precision;
    m_h_iq = h_iq;
    m_h_ik = h_ik;
    m_h_sel = h_sel;
    m_batch = batch;
    m_scenario = scenario;
    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    if (data_precision == ov::element::bf16) {
        rel_threshold = 5e-2F;
        abs_threshold = 5e-2F;
    } else if (data_precision == ov::element::f16) {
        rel_threshold = 5e-3F;
        abs_threshold = 5e-3F;
    }

    m_didx = 16;
    m_rotary_dim = 8;
    m_block_topk = 2;
    m_calls.clear();

    if (m_scenario == QSAIndexerScenario::kWorkedExample) {
        m_batch = 2;
        m_compress_ratio = 4;
        m_summary_cap = 3;
        m_calls.push_back(CallSpec{10, {10, 7}});
        m_calls.push_back(CallSpec{1, {11, 8}});
    } else if (m_scenario == QSAIndexerScenario::kFullyPaddedRow) {
        m_batch = 2;
        m_compress_ratio = 4;
        m_summary_cap = 3;
        m_calls.push_back(CallSpec{5, {5, 0}});
        m_calls.push_back(CallSpec{3, {8, 3}});
    } else if (m_scenario == QSAIndexerScenario::kRaggedCalls) {
        m_batch = 2;
        m_compress_ratio = 4;
        m_summary_cap = 4;
        m_calls.push_back(CallSpec{7, {7, 5}});
        m_calls.push_back(CallSpec{3, {10, 8}});
        m_calls.push_back(CallSpec{5, {15, 13}});
    } else {
        m_compress_ratio = 2;
        m_summary_cap = 4;
        std::vector<int32_t> key_length(m_batch, 7);
        m_calls.push_back(CallSpec{7, key_length});
    }

    auto p_q_idx = std::make_shared<ov::op::v0::Parameter>(m_data_precision,
                                                           ov::PartialShape{static_cast<int64_t>(m_batch),
                                                                            -1,
                                                                            static_cast<int64_t>(m_h_iq),
                                                                            static_cast<int64_t>(m_didx)});
    auto p_k_idx = std::make_shared<ov::op::v0::Parameter>(m_data_precision,
                                                           ov::PartialShape{static_cast<int64_t>(m_batch),
                                                                            -1,
                                                                            static_cast<int64_t>(m_h_ik),
                                                                            static_cast<int64_t>(m_didx)});
    auto p_rope_cos = std::make_shared<ov::op::v0::Parameter>(
        m_data_precision,
        ov::PartialShape{static_cast<int64_t>(m_batch), -1, static_cast<int64_t>(m_rotary_dim)});
    auto p_rope_sin = std::make_shared<ov::op::v0::Parameter>(
        m_data_precision,
        ov::PartialShape{static_cast<int64_t>(m_batch), -1, static_cast<int64_t>(m_rotary_dim)});
    auto p_key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_batch});
    auto p_q_norm_weight = std::make_shared<ov::op::v0::Parameter>(m_data_precision, ov::Shape{m_didx});
    auto p_k_norm_weight = std::make_shared<ov::op::v0::Parameter>(m_data_precision, ov::Shape{m_didx});
    // Only the blocks of the earlier calls: the op returns them grown by the blocks of this call.
    auto p_summary_cache = std::make_shared<ov::op::v0::Parameter>(m_data_precision,
                                                                   ov::PartialShape{static_cast<int64_t>(m_batch),
                                                                                    static_cast<int64_t>(m_h_ik),
                                                                                    -1,
                                                                                    static_cast<int64_t>(m_didx)});
    auto p_past_raw_k =
        std::make_shared<ov::op::v0::Parameter>(m_data_precision,
                                                ov::Shape{m_batch, m_h_ik, m_compress_ratio - 1, m_didx});
    auto p_past_rope = std::make_shared<ov::op::v0::Parameter>(m_data_precision, ov::Shape{m_batch, 2 * m_rotary_dim});

    p_q_idx->set_friendly_name("q_idx");
    p_k_idx->set_friendly_name("k_idx");
    p_rope_cos->set_friendly_name("rope_cos");
    p_rope_sin->set_friendly_name("rope_sin");
    p_key_length->set_friendly_name("key_length");
    p_q_norm_weight->set_friendly_name("q_norm_weight");
    p_k_norm_weight->set_friendly_name("k_norm_weight");
    p_summary_cache->set_friendly_name("summary_cache");
    p_past_raw_k->set_friendly_name("past_raw_k");
    p_past_rope->set_friendly_name("past_rope");

    ov::intel_cpu::QSAIndexer::Config config;
    config.compress_ratio = m_compress_ratio;
    config.block_topk = m_block_topk;
    config.h_sel = m_h_sel;
    config.indexer_rotary_dim = m_rotary_dim;
    config.indexer_head_dim = m_didx;
    config.eps = 1e-6F;
    config.scale = 0.25F;

    ov::OutputVector args = {p_q_idx,
                             p_k_idx,
                             p_rope_cos,
                             p_rope_sin,
                             p_key_length,
                             p_q_norm_weight,
                             p_k_norm_weight,
                             p_summary_cache,
                             p_past_raw_k,
                             p_past_rope};
    using OutputIdx = ov::intel_cpu::QSAIndexer::OutputIdx;
    auto qsa_indexer_v2 = std::make_shared<ov::intel_cpu::QSAIndexer>(args, config);

    function = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(qsa_indexer_v2->output(OutputIdx::SEL_INDICES)),
                         std::make_shared<ov::op::v0::Result>(qsa_indexer_v2->output(OutputIdx::SEL_COUNT)),
                         std::make_shared<ov::op::v0::Result>(qsa_indexer_v2->output(OutputIdx::SUMMARY_CACHE)),
                         std::make_shared<ov::op::v0::Result>(qsa_indexer_v2->output(OutputIdx::PAST_INCOMPLETE_RAW_K)),
                         std::make_shared<ov::op::v0::Result>(qsa_indexer_v2->output(OutputIdx::PAST_INCOMPLETE_ROPE))},
        ov::ParameterVector{p_q_idx,
                            p_k_idx,
                            p_rope_cos,
                            p_rope_sin,
                            p_key_length,
                            p_q_norm_weight,
                            p_k_norm_weight,
                            p_summary_cache,
                            p_past_raw_k,
                            p_past_rope});
}

void QSAIndexerLayerCPUTest::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

std::vector<ov::Tensor> QSAIndexerLayerCPUTest::run_call(const CallSpec& call, size_t call_idx) {
    const size_t L = static_cast<size_t>(call.length);

    ov::Tensor t_q_idx = ov::test::utils::create_and_fill_tensor(
        m_data_precision,
        ov::Shape{m_batch, L, m_h_iq, m_didx},
        ov::test::utils::InputGenerateData(-1.0, 2000, 1000, static_cast<int32_t>(10 + call_idx * 4 + 1)));
    ov::Tensor t_k_idx = ov::test::utils::create_and_fill_tensor(
        m_data_precision,
        ov::Shape{m_batch, L, m_h_ik, m_didx},
        ov::test::utils::InputGenerateData(-1.0, 2000, 1000, static_cast<int32_t>(10 + call_idx * 4 + 2)));

    // rope_cos/rope_sin can't be arbitrary random data: they must be genuine (cos, sin) pairs,
    // pre-duplicated across both halves of rotary_dim (per new_spec's rotate-half convention), so
    // they're synthesized directly rather than through create_and_fill_tensor.
    std::vector<float> rope_cos_data(m_batch * L * m_rotary_dim);
    std::vector<float> rope_sin_data(m_batch * L * m_rotary_dim);
    const size_t half = m_rotary_dim / 2;
    for (size_t b = 0; b < m_batch; b++) {
        for (size_t l = 0; l < L; l++) {
            const int64_t abs_pos = call.key_length[b] - call.length + static_cast<int64_t>(l);
            for (size_t j = 0; j < m_rotary_dim; j++) {
                const size_t jj = j % half;
                const float theta = static_cast<float>(abs_pos) * 0.1F + static_cast<float>(jj) * 0.05F;
                rope_cos_data[(b * L + l) * m_rotary_dim + j] = std::cos(theta);
                rope_sin_data[(b * L + l) * m_rotary_dim + j] = std::sin(theta);
            }
        }
    }
    ov::Tensor t_rope_cos(m_data_precision, ov::Shape{m_batch, L, m_rotary_dim});
    ov::Tensor t_rope_sin(m_data_precision, ov::Shape{m_batch, L, m_rotary_dim});
    write_f32_to_tensor(t_rope_cos, rope_cos_data);
    write_f32_to_tensor(t_rope_sin, rope_sin_data);

    ov::Tensor t_key_length(ov::element::i32, ov::Shape{m_batch});
    std::copy(call.key_length.begin(), call.key_length.end(), t_key_length.data<int32_t>());

    const std::unordered_map<std::string, ov::Tensor> name_to_tensor = {
        {"q_idx", t_q_idx},
        {"k_idx", t_k_idx},
        {"rope_cos", t_rope_cos},
        {"rope_sin", t_rope_sin},
        {"key_length", t_key_length},
        {"q_norm_weight", m_q_norm_weight_tensor},
        {"k_norm_weight", m_k_norm_weight_tensor},
        {"summary_cache", m_summary_cache_tensor},
        {"past_raw_k", m_past_raw_k_tensor},
        {"past_rope", m_past_rope_tensor},
    };
    for (const auto& port : compiledModel.inputs()) {
        const auto it = name_to_tensor.find(port.get_node()->get_friendly_name());
        if (it != name_to_tensor.end()) {
            m_infer_request.set_tensor(port, it->second);
        }
    }

    // The op must not update its state inputs in place: snapshot them to check afterwards.
    std::vector<ov::Tensor> state_in_snapshot;
    for (const auto* state : {&m_summary_cache_tensor, &m_past_raw_k_tensor, &m_past_rope_tensor}) {
        ov::Tensor snapshot{state->get_element_type(), state->get_shape()};
        state->copy_to(snapshot);
        state_in_snapshot.push_back(std::move(snapshot));
    }

    m_infer_request.infer();

    std::vector<ov::Tensor> outs;
    for (const auto& port : compiledModel.outputs()) {
        const auto& src = m_infer_request.get_tensor(port);
        ov::Tensor c{src.get_element_type(), src.get_shape()};
        src.copy_to(c);
        outs.push_back(std::move(c));
    }

    QSAIndexerOracle oracle;
    oracle.h_iq = m_h_iq;
    oracle.h_ik = m_h_ik;
    oracle.h_sel = m_h_sel;
    oracle.d_idx = m_didx;
    oracle.rotary_dim = m_rotary_dim;
    oracle.r = static_cast<int64_t>(m_compress_ratio);
    oracle.block_topk = m_block_topk;
    oracle.eps = 1e-6F;
    oracle.scale = 0.25F;
    oracle.summary_cap = m_summary_cap;
    oracle.summary_cache = std::move(m_oracle_summary_cache);
    oracle.past_raw_k = std::move(m_oracle_past_raw_k);
    oracle.past_rope = std::move(m_oracle_past_rope);

    auto expected = oracle.run_call(m_batch,
                                    call.length,
                                    call.key_length,
                                    read_tensor_to_f32(t_q_idx),
                                    read_tensor_to_f32(t_k_idx),
                                    read_tensor_to_f32(t_rope_cos),
                                    read_tensor_to_f32(t_rope_sin),
                                    read_tensor_to_f32(m_q_norm_weight_tensor),
                                    read_tensor_to_f32(m_k_norm_weight_tensor));
    m_oracle_summary_cache = std::move(oracle.summary_cache);
    m_oracle_past_raw_k = std::move(oracle.past_raw_k);
    m_oracle_past_rope = std::move(oracle.past_rope);

    ov::Tensor expected_sel_indices(ov::element::i32, ov::Shape{m_batch, m_h_sel, L, m_block_topk});
    ov::Tensor expected_sel_count(ov::element::i32, ov::Shape{m_batch, m_h_sel, L});
    std::copy(expected.sel_indices.begin(), expected.sel_indices.end(), expected_sel_indices.data<int32_t>());
    std::copy(expected.sel_count.begin(), expected.sel_count.end(), expected_sel_count.data<int32_t>());

    // The output holds the blocks of the longest row: max(key_length) / compress_ratio.
    const size_t num_blocks = static_cast<size_t>(*std::max_element(call.key_length.begin(), call.key_length.end()) /
                                                  static_cast<int64_t>(m_compress_ratio));
    ov::Tensor expected_summary_cache(m_data_precision, ov::Shape{m_batch, m_h_ik, num_blocks, m_didx});
    std::vector<float> expected_summary(m_batch * m_h_ik * num_blocks * m_didx);
    for (size_t bh = 0; bh < m_batch * m_h_ik; bh++) {
        std::copy_n(m_oracle_summary_cache.begin() + bh * m_summary_cap * m_didx,
                    num_blocks * m_didx,
                    expected_summary.begin() + bh * num_blocks * m_didx);
    }
    ov::Tensor expected_past_raw_k(m_data_precision, m_past_raw_k_tensor.get_shape());
    ov::Tensor expected_past_rope(m_data_precision, m_past_rope_tensor.get_shape());
    write_f32_to_tensor(expected_summary_cache, expected_summary);
    write_f32_to_tensor(expected_past_raw_k, m_oracle_past_raw_k);
    write_f32_to_tensor(expected_past_rope, m_oracle_past_rope);

    compare({expected_sel_indices, expected_sel_count, expected_summary_cache, expected_past_raw_k, expected_past_rope},
            outs);

    // State inputs are untouched (bitwise)...
    const std::vector<const ov::Tensor*> state_in = {&m_summary_cache_tensor,
                                                     &m_past_raw_k_tensor,
                                                     &m_past_rope_tensor};
    for (size_t i = 0; i < state_in.size(); i++) {
        if (state_in_snapshot[i].get_byte_size() == 0) {
            continue;
        }
        EXPECT_EQ(0,
                  std::memcmp(state_in[i]->data(), state_in_snapshot[i].data(), state_in_snapshot[i].get_byte_size()))
            << "state input " << i << " was modified in place";
    }
    // ...and the state outputs are what the outside Assign feeds into the next call.
    m_summary_cache_tensor = outs[ov::intel_cpu::QSAIndexer::OutputIdx::SUMMARY_CACHE];
    m_past_raw_k_tensor = outs[ov::intel_cpu::QSAIndexer::OutputIdx::PAST_INCOMPLETE_RAW_K];
    m_past_rope_tensor = outs[ov::intel_cpu::QSAIndexer::OutputIdx::PAST_INCOMPLETE_ROPE];
    return outs;
}

void QSAIndexerLayerCPUTest::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    compiledModel = core->compile_model(function, targetDevice, configuration);
    m_infer_request = compiledModel.create_infer_request();

    m_summary_cache_tensor = make_zero_tensor(m_data_precision, ov::Shape{m_batch, m_h_ik, 0, m_didx});
    m_past_raw_k_tensor = make_zero_tensor(m_data_precision, ov::Shape{m_batch, m_h_ik, m_compress_ratio - 1, m_didx});
    m_past_rope_tensor = make_zero_tensor(m_data_precision, ov::Shape{m_batch, 2 * m_rotary_dim});

    m_q_norm_weight_tensor =
        ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                ov::Shape{m_didx},
                                                ov::test::utils::InputGenerateData(0.5, 1000, 1000, 6));
    m_k_norm_weight_tensor =
        ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                ov::Shape{m_didx},
                                                ov::test::utils::InputGenerateData(0.5, 1000, 1000, 7));

    m_oracle_summary_cache.assign(m_batch * m_h_ik * m_summary_cap * m_didx, 0.0F);
    m_oracle_past_raw_k.assign(m_batch * m_h_ik * (m_compress_ratio - 1) * m_didx, 0.0F);
    m_oracle_past_rope.assign(m_batch * 2 * m_rotary_dim, 0.0F);

    for (size_t i = 0; i < m_calls.size(); i++) {
        run_call(m_calls[i], i);
    }
}

TEST_P(QSAIndexerLayerCPUTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
