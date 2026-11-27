// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_sdpa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "utils/precision_support.h"
#include "utils/qsa_indexer_reference.hpp"

namespace {

// Host-vector oracle of the non-paged consumer spec: the causal tail block is derived here, not taken from sel_indices.
std::vector<float> run_sparse_sdpa_reference(size_t hq,
                                             size_t hkv,
                                             size_t dv,
                                             size_t len,
                                             size_t kmax,
                                             size_t batch,
                                             size_t h_sel,
                                             bool causal,
                                             size_t k_head_size,
                                             size_t sel_block_size,
                                             float scale,
                                             const std::vector<float>& query,
                                             const std::vector<float>& key,
                                             const std::vector<float>& value,
                                             const std::vector<int32_t>& sel_indices,
                                             const std::vector<int32_t>& sel_count,
                                             const std::vector<size_t>& key_length) {
    auto key_at = [&](size_t b, size_t head, size_t pos) {
        return &key[((b * hkv + head) * len + pos) * k_head_size];
    };
    auto value_at = [&](size_t b, size_t head, size_t pos) {
        return &value[((b * hkv + head) * len + pos) * dv];
    };

    std::vector<float> output(batch * hq * len * dv, 0.0F);
    const size_t group = hq / hkv;
    const size_t q_per_sel = hq / h_sel;
    for (size_t b = 0; b < batch; b++) {
        // pad = physically-empty rows at the front of batch b's dense [Hkv, S, D] K/V history;
        // key_length empty means no padding (pad == 0 for every batch), matching today's behavior.
        const int64_t pad = key_length.empty() ? 0 : static_cast<int64_t>(len) - static_cast<int64_t>(key_length[b]);
        for (size_t l = 0; l < len; l++) {
            const int64_t token_pos = static_cast<int64_t>(l);
            for (size_t h = 0; h < hq; h++) {
                const size_t kv_head = h / group;
                const size_t sh = h / q_per_sel;
                const float* q = &query[((b * hq + h) * len + l) * k_head_size];
                std::vector<float> acc(dv, 0.0F);
                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0F;
                // One online-softmax step for key/value position `kpos`: score = (q . K) * scale, rescale the running
                // max/sum by corr, then acc = acc * corr + p * V. Positions outside the valid history are skipped.
                auto attend_token = [&](int64_t kpos) {
                    if (kpos < pad || kpos >= static_cast<int64_t>(len)) {
                        return;
                    }
                    if (causal && kpos > token_pos) {
                        return;
                    }
                    const float* k = key_at(b, kv_head, static_cast<size_t>(kpos));
                    const float* v = value_at(b, kv_head, static_cast<size_t>(kpos));
                    float dot = 0.0F;
                    for (size_t d = 0; d < k_head_size; d++) {
                        dot += q[d] * k[d];
                    }
                    const float score = dot * scale;
                    const float new_max = std::max(running_max, score);
                    const float corr = std::exp(running_max - new_max);
                    const float p = std::exp(score - new_max);
                    running_sum = running_sum * corr + p;
                    for (size_t d = 0; d < dv; d++) {
                        acc[d] = acc[d] * corr + p * v[d];
                    }
                    running_max = new_max;
                };

                const size_t nblocks = static_cast<size_t>(sel_count[(b * h_sel + sh) * len + l]);
                for (size_t i = 0; i < nblocks; i++) {
                    const int64_t block_id = sel_indices[((b * h_sel + sh) * len + l) * kmax + i];
                    const int64_t block_start = pad + block_id * static_cast<int64_t>(sel_block_size);
                    for (int64_t p = block_start; p < block_start + static_cast<int64_t>(sel_block_size); p++) {
                        attend_token(p);
                    }
                }

                // sel_indices / sel_count only carry complete blocks; when causal the consumer derives the current
                // block's tail itself.
                if (causal) {
                    const int64_t tail_start = pad + (((token_pos - pad + 1) / static_cast<int64_t>(sel_block_size)) *
                                                      static_cast<int64_t>(sel_block_size));
                    for (int64_t p = tail_start; p <= token_pos; p++) {
                        attend_token(p);
                    }
                }

                const float inv_sum = running_sum > 0.0F ? 1.0F / running_sum : 0.0F;
                for (size_t d = 0; d < dv; d++) {
                    output[((b * hq + h) * len + l) * dv + d] = acc[d] * inv_sum;
                }
            }
        }
    }
    return output;
}

}  // namespace

namespace ov {
namespace test {

std::string SparseSDPALayerCPUTest::getTestCaseName(const testing::TestParamInfo<SparseSDPALayerTestParams>& obj) {
    const auto& [batch,
                 num_q_heads,
                 num_kv_heads,
                 k_head_size,
                 v_head_size,
                 seq_len,
                 h_sel,
                 sel_block_size,
                 max_blocks,
                 block_step,
                 causal,
                 shuffle,
                 key_length,
                 data_precision,
                 device] = obj.param;
    std::ostringstream result;
    result << "Batch=" << batch << "_Hq=" << num_q_heads << "_Hkv=" << num_kv_heads << "_Dh=" << k_head_size
           << "_Dv=" << v_head_size << "_Len=" << seq_len << "_Hsel=" << h_sel << "_Bk=" << sel_block_size
           << "_Kmax=" << max_blocks << "_Step=" << block_step << "_Causal=" << causal << "_Shuffled=" << shuffle
           << "_KeyLength=";
    for (size_t i = 0; i < key_length.size(); i++) {
        result << (i > 0 ? "x" : "") << key_length[i];
    }
    result << "_DataPrecision=" << data_precision << "_Target=" << device;
    return result.str();
}

void SparseSDPALayerCPUTest::SetUp() {
    const auto& [batch,
                 num_q_heads,
                 num_kv_heads,
                 k_head_size,
                 v_head_size,
                 seq_len,
                 h_sel,
                 sel_block_size,
                 max_blocks,
                 block_step,
                 causal,
                 shuffle,
                 key_length,
                 data_precision,
                 device] = GetParam();

    if (!ov::intel_cpu::hasHardwareSupport(data_precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << data_precision << " data on this system.";
    }
    OPENVINO_ASSERT(num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, "Hkv must divide Hq");
    OPENVINO_ASSERT(key_length.empty() || key_length.size() == batch, "one key_length entry per batch item");
    targetDevice = device;
    m_data_precision = data_precision;
    m_batch = batch;
    m_num_q_heads = num_q_heads;
    m_num_kv_heads = num_kv_heads;
    m_k_head_size = k_head_size;
    m_v_head_size = v_head_size;
    m_seq_len = seq_len;
    m_h_sel = h_sel;
    m_sel_block_size = sel_block_size;
    m_max_blocks = max_blocks;
    m_block_step = block_step;
    m_causal = causal;
    m_shuffle = shuffle;
    m_key_length = key_length;
    m_scale = 1.0F / std::sqrt(static_cast<float>(m_k_head_size));
    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    if (data_precision == ov::element::bf16) {
        rel_threshold = 1e-2F;
        abs_threshold = 1e-2F;
    } else if (data_precision == ov::element::f16) {
        rel_threshold = 1e-3F;
        abs_threshold = 1e-3F;
    }

    // The sequence length is dynamic in the model, its target shape is the seq_len parameter.
    const auto dim = [](size_t value) {
        return static_cast<ov::Dimension::value_type>(value);
    };
    const auto dynamic = ov::Dimension::dynamic();
    const ov::PartialShape query_shape{dim(batch), dim(num_q_heads), dynamic, dim(k_head_size)};
    const ov::PartialShape key_shape{dim(batch), dim(num_kv_heads), dynamic, dim(k_head_size)};
    const ov::PartialShape value_shape{dim(batch), dim(num_kv_heads), dynamic, dim(v_head_size)};
    const ov::PartialShape sel_indices_shape{dim(batch), dim(h_sel), dynamic, dim(max_blocks)};
    const ov::PartialShape sel_count_shape{dim(batch), dim(h_sel), dynamic};

    std::vector<InputShape> input_shapes{{query_shape, {ov::Shape{batch, num_q_heads, seq_len, k_head_size}}},
                                         {key_shape, {ov::Shape{batch, num_kv_heads, seq_len, k_head_size}}},
                                         {value_shape, {ov::Shape{batch, num_kv_heads, seq_len, v_head_size}}},
                                         {sel_indices_shape, {ov::Shape{batch, h_sel, seq_len, max_blocks}}},
                                         {sel_count_shape, {ov::Shape{batch, h_sel, seq_len}}}};

    auto p_query = std::make_shared<ov::op::v0::Parameter>(data_precision, query_shape);
    auto p_key = std::make_shared<ov::op::v0::Parameter>(data_precision, key_shape);
    auto p_value = std::make_shared<ov::op::v0::Parameter>(data_precision, value_shape);
    auto p_sel_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, sel_indices_shape);
    auto p_sel_count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, sel_count_shape);
    ov::ParameterVector params = {p_query, p_key, p_value, p_sel_indices, p_sel_count};
    ov::OutputVector args = {p_query, p_key, p_value, p_sel_indices, p_sel_count};

    if (!m_key_length.empty()) {
        const ov::PartialShape key_length_shape{dim(batch)};
        input_shapes.emplace_back(key_length_shape, std::vector<ov::Shape>{ov::Shape{batch}});
        auto p_key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, key_length_shape);
        params.push_back(p_key_length);
        args.push_back(p_key_length);
    }

    init_input_shapes(input_shapes);

    auto sparse_sdpa = std::make_shared<ov::op::v17::SparseAttention>(args,
                                                                      static_cast<int64_t>(m_sel_block_size),
                                                                      /*query_block_size=*/1,
                                                                      m_causal);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sparse_sdpa)}, params);
}

void SparseSDPALayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    // The sequence length of this inference is the one of the query target shape.
    m_seq_len = targetInputStaticShapes.at(0).at(2);

    const auto make_data = [this](const ov::Shape& shape, uint32_t seed) {
        return ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                       shape,
                                                       ov::test::utils::InputGenerateData(-1, 2, 10000, seed));
    };
    const auto make_index = [](const std::vector<int32_t>& values, const ov::Shape& shape) {
        ov::Tensor t(ov::element::i32, shape);
        std::copy(values.begin(), values.end(), t.data<int32_t>());
        return t;
    };

    const auto& params = function->get_parameters();
    std::vector<ov::Tensor> tensors(params.size());
    tensors[0] = make_data(ov::Shape{m_batch, m_num_q_heads, m_seq_len, m_k_head_size}, 1);
    // Different values for every batch item, so a kernel bug that reused batch 0's cache slot for batch 1 produces
    // numerically different (and thus detectable) output.
    tensors[1] = make_data(ov::Shape{m_batch, m_num_kv_heads, m_seq_len, m_k_head_size}, 2);
    tensors[2] = make_data(ov::Shape{m_batch, m_num_kv_heads, m_seq_len, m_v_head_size}, 3);

    // Synthetic selection: up to max_blocks complete blocks before the query's block, block_step apart, ascending or
    // shuffled. Block ids count from the non-padded history: with padding the query's block is (l - pad) / bk.
    std::mt19937 shuffle_gen(2026);
    std::vector<int32_t> sel_indices_flat(m_batch * m_h_sel * m_seq_len * m_max_blocks, -1);
    std::vector<int32_t> sel_count_flat(m_batch * m_h_sel * m_seq_len, 0);
    for (size_t b = 0; b < m_batch; b++) {
        const int64_t pad =
            m_key_length.empty() ? 0 : static_cast<int64_t>(m_seq_len) - static_cast<int64_t>(m_key_length[b]);
        for (size_t l = 0; l < m_seq_len; l++) {
            const int64_t rel_pos = static_cast<int64_t>(l) - pad;
            const int32_t cur_block =
                rel_pos >= 0 ? static_cast<int32_t>(rel_pos / static_cast<int64_t>(m_sel_block_size)) : 0;
            const size_t num_sel =
                rel_pos >= 0 ? std::min(static_cast<size_t>(cur_block) / m_block_step, m_max_blocks) : 0;
            for (size_t sh = 0; sh < m_h_sel; sh++) {
                const size_t base = (b * m_h_sel + sh) * m_seq_len + l;
                for (size_t k = 0; k < num_sel; k++) {
                    sel_indices_flat[base * m_max_blocks + k] =
                        cur_block - static_cast<int32_t>(m_block_step * (num_sel - k));
                }
                if (m_shuffle) {
                    auto* row = sel_indices_flat.data() + base * m_max_blocks;
                    std::shuffle(row, row + num_sel, shuffle_gen);
                }
                sel_count_flat[base] = static_cast<int32_t>(num_sel);
            }
        }
    }
    tensors[3] = make_index(sel_indices_flat, ov::Shape{m_batch, m_h_sel, m_seq_len, m_max_blocks});
    tensors[4] = make_index(sel_count_flat, ov::Shape{m_batch, m_h_sel, m_seq_len});

    if (!m_key_length.empty()) {
        std::vector<int32_t> key_length_flat(m_key_length.size());
        std::transform(m_key_length.begin(), m_key_length.end(), key_length_flat.begin(), [](size_t v) {
            return static_cast<int32_t>(v);
        });
        tensors[5] = make_index(key_length_flat, ov::Shape{m_batch});
    }

    for (size_t i = 0; i < params.size(); i++) {
        host_inputs[params[i]] = tensors[i];
        inputs[params[i]] = tensors[i];
    }
}

std::vector<ov::Tensor> SparseSDPALayerCPUTest::calculate_refs() {
    const auto& params = function->get_parameters();
    const auto to_f32 = [&](const std::shared_ptr<ov::Node>& p) {
        return read_tensor_to_f32(host_inputs.at(p));
    };
    const auto to_i32 = [&](const std::shared_ptr<ov::Node>& p) {
        const auto& t = host_inputs.at(p);
        const auto* ptr = t.data<const int32_t>();
        return std::vector<int32_t>(ptr, ptr + t.get_size());
    };

    auto query = to_f32(params[0]);
    auto key = to_f32(params[1]);
    auto value = to_f32(params[2]);
    auto sel_indices = to_i32(params[3]);
    auto sel_count = to_i32(params[4]);

    auto output = run_sparse_sdpa_reference(m_num_q_heads,
                                            m_num_kv_heads,
                                            m_v_head_size,
                                            m_seq_len,
                                            m_max_blocks,
                                            m_batch,
                                            m_h_sel,
                                            m_causal,
                                            m_k_head_size,
                                            m_sel_block_size,
                                            m_scale,
                                            query,
                                            key,
                                            value,
                                            sel_indices,
                                            sel_count,
                                            m_key_length);

    ov::Tensor output_tensor(ov::element::f32, ov::Shape{m_batch, m_num_q_heads, m_seq_len, m_v_head_size});
    std::copy(output.begin(), output.end(), output_tensor.data<float>());
    return {output_tensor};
}

void SparseSDPALayerCPUTest::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

TEST_P(SparseSDPALayerCPUTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
