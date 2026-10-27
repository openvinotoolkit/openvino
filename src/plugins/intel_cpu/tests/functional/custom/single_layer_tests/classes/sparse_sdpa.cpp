// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_sdpa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sparse_attention.hpp"

namespace {

// Fixed configuration of this test: B independent batch items, each with its own dense [Hkv, S, D] K/V history and L
// query positions. S == L (see SparseSDPAShape), so the bottom-right-aligned logical position q_abs = S - L + l reduces
// to q_abs == l, matching a straightforward causal prefill over the whole history. k_head_size / sel_block_size are
// parameterized (see SparseSDPALayerTestParams) so the AMX paths -- which require k_head_size % 32 == 0, merging
// adjacent selected blocks up to a 16-row multiple when sel_block_size itself isn't one (see m_amx_unit_blocks in
// sparse_sdpa.cpp)
// -- can be exercised by dedicated instantiations. The synthetic selection below selects up to max_blocks complete
// blocks before the query's own block, block_step apart (ascending, duplicate-free, matching the real producer
// contract): with step 1 the query rows near the end of the sequence build up a long run of physically-adjacent
// selected blocks (L = 40 and K_max = 10 give the sel_block_size == 4 AMX instantiation a run of up to 9 adjacent
// blocks), with step 2 there are none.
constexpr size_t kHq = 4;
constexpr size_t kHkv = 2;
constexpr size_t kDv = 16;

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
// spec, used purely as a test oracle -- NOT a copy of the CPU node's PlainTensor-based indexing.
// `query`/`key`/`value` are plain batched [B, H, {L|S}, D]; `sel_indices`/`sel_count` are
// [B, h_sel, L, Kmax] / [B, h_sel, L]. Since S == L, the bottom-right-aligned logical
// position q_abs = S - L + l reduces to q_abs == l. When causal, the current (possibly
// incomplete) block is never in sel_indices/sel_count -- this oracle derives and attends it
// itself, matching the CPU node's own automatic tail derivation.
std::vector<float> run_sparse_sdpa_reference(size_t len,
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
        return &key[((b * kHkv + head) * len + pos) * k_head_size];
    };
    auto value_at = [&](size_t b, size_t head, size_t pos) {
        return &value[((b * kHkv + head) * len + pos) * kDv];
    };

    std::vector<float> output(batch * kHq * len * kDv, 0.0F);
    const size_t group = kHq / kHkv;
    const size_t q_per_sel = kHq / h_sel;
    for (size_t b = 0; b < batch; b++) {
        // pad = physically-empty rows at the front of batch b's dense [Hkv, S, D] K/V history;
        // key_length empty means no padding (pad == 0 for every batch), matching today's behavior.
        const int64_t pad = key_length.empty() ? 0 : static_cast<int64_t>(len) - static_cast<int64_t>(key_length[b]);
        for (size_t l = 0; l < len; l++) {
            const int64_t token_pos = static_cast<int64_t>(l);
            for (size_t h = 0; h < kHq; h++) {
                const size_t kv_head = h / group;
                const size_t sh = h / q_per_sel;
                const float* q = &query[((b * kHq + h) * len + l) * k_head_size];
                std::vector<float> acc(kDv, 0.0F);
                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0F;
                auto consume = [&](int64_t kpos) {
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
                    for (size_t d = 0; d < kDv; d++) {
                        acc[d] = acc[d] * corr + p * v[d];
                    }
                    running_max = new_max;
                };

                const size_t nblocks = static_cast<size_t>(sel_count[(b * h_sel + sh) * len + l]);
                for (size_t i = 0; i < nblocks; i++) {
                    const int64_t block_id = sel_indices[((b * h_sel + sh) * len + l) * kmax + i];
                    const int64_t block_start = pad + block_id * static_cast<int64_t>(sel_block_size);
                    for (int64_t p = block_start; p < block_start + static_cast<int64_t>(sel_block_size); p++) {
                        consume(p);
                    }
                }

                // sel_indices/sel_count only ever carry *complete* selected blocks -- the current
                // (possibly incomplete) block is never in them. When causal, the consumer derives
                // and attends that tail itself from the query's own logical position.
                if (causal) {
                    const int64_t tail_start =
                        pad + (((token_pos - pad + 1) / static_cast<int64_t>(sel_block_size)) *
                               static_cast<int64_t>(sel_block_size));
                    for (int64_t p = tail_start; p <= token_pos; p++) {
                        consume(p);
                    }
                }

                const float inv_sum = running_sum > 0.0F ? 1.0F / running_sum : 0.0F;
                for (size_t d = 0; d < kDv; d++) {
                    output[((b * kHq + h) * len + l) * kDv + d] = acc[d] * inv_sum;
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
    const auto& [device, data_precision, h_sel, causal, batch, k_head_size, sel_block_size, key_length, shape] =
        obj.param;
    std::string name = "Target=" + device + "_DataPrecision=" + data_precision.get_type_name() +
                       "_Hsel=" + std::to_string(h_sel) + "_Causal=" + std::to_string(causal) +
                       "_Batch=" + std::to_string(batch) + "_Dh=" + std::to_string(k_head_size) +
                       "_Bk=" + std::to_string(sel_block_size) + "_Len=" + std::to_string(shape.seq_len) +
                       "_Kmax=" + std::to_string(shape.max_blocks) + "_Step=" + std::to_string(shape.block_step);
    if (!key_length.empty()) {
        name += "_KeyLength";
        for (auto len : key_length) {
            name += "_" + std::to_string(len);
        }
    }
    return name;
}

void SparseSDPALayerCPUTest::SetUp() {
    const auto& [device, data_precision, h_sel, causal, batch, k_head_size, sel_block_size, key_length, shape] =
        GetParam();
    targetDevice = device;
    m_data_precision = data_precision;
    m_h_sel = h_sel;
    m_causal = causal;
    m_batch = batch;
    m_k_head_size = k_head_size;
    m_sel_block_size = sel_block_size;
    m_key_length = key_length;
    m_shape = shape;
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

    auto p_query = std::make_shared<ov::op::v0::Parameter>(data_precision,
                                                           ov::Shape{m_batch, kHq, m_shape.seq_len, m_k_head_size});
    auto p_key = std::make_shared<ov::op::v0::Parameter>(data_precision,
                                                         ov::Shape{m_batch, kHkv, m_shape.seq_len, m_k_head_size});
    auto p_value =
        std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{m_batch, kHkv, m_shape.seq_len, kDv});
    auto p_sel_indices =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                ov::Shape{m_batch, m_h_sel, m_shape.seq_len, m_shape.max_blocks});
    auto p_sel_count =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_batch, m_h_sel, m_shape.seq_len});

    std::vector<ov::Shape> input_shapes = {p_query->get_shape(),
                                           p_key->get_shape(),
                                           p_value->get_shape(),
                                           p_sel_indices->get_shape(),
                                           p_sel_count->get_shape()};
    ov::ParameterVector params = {p_query, p_key, p_value, p_sel_indices, p_sel_count};
    ov::OutputVector args = {p_query, p_key, p_value, p_sel_indices, p_sel_count};

    std::shared_ptr<ov::op::v0::Parameter> p_key_length;
    if (!m_key_length.empty()) {
        p_key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_batch});
        input_shapes.push_back(p_key_length->get_shape());
        params.push_back(p_key_length);
        args.push_back(p_key_length);
    }

    init_input_shapes(static_shapes_to_test_representation(input_shapes));

    auto sparse_sdpa = std::make_shared<ov::op::v17::SparseAttention>(args,
                                                                      static_cast<int64_t>(m_sel_block_size),
                                                                      /*query_block_size=*/1,
                                                                      m_causal);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sparse_sdpa)}, params);
}

void SparseSDPALayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
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
    const auto make_data = [this, &assign_precision](const ov::Shape& shape, uint32_t seed) {
        ov::Tensor t(m_data_precision, shape);
        std::vector<float> data(t.get_size());
        fill_pseudo_random(data, seed);
        assign_precision(t, data);
        return t;
    };
    const auto make_index = [](const std::vector<int32_t>& values, const ov::Shape& shape) {
        ov::Tensor t(ov::element::i32, shape);
        std::copy(values.begin(), values.end(), t.data<int32_t>());
        return t;
    };

    const auto& params = function->get_parameters();
    std::vector<ov::Tensor> tensors(params.size());
    tensors[0] = make_data(ov::Shape{m_batch, kHq, m_shape.seq_len, m_k_head_size}, 1);
    {
        // Each batch slot gets a distinct seed so a kernel bug that reused batch 0's cache slot
        // for batch 1 produces numerically different (and thus detectable) output.
        ov::Tensor key(m_data_precision, ov::Shape{m_batch, kHkv, m_shape.seq_len, m_k_head_size});
        ov::Tensor value(m_data_precision, ov::Shape{m_batch, kHkv, m_shape.seq_len, kDv});
        const size_t key_per_batch = kHkv * m_shape.seq_len * m_k_head_size;
        const size_t value_per_batch = kHkv * m_shape.seq_len * kDv;
        std::vector<float> key_data(key.get_size());
        std::vector<float> value_data(value.get_size());
        for (size_t b = 0; b < m_batch; b++) {
            std::vector<float> key_slot(key_per_batch);
            std::vector<float> value_slot(value_per_batch);
            fill_pseudo_random(key_slot, 2 + static_cast<uint32_t>(b) * 11U);
            fill_pseudo_random(value_slot, 3 + static_cast<uint32_t>(b) * 11U);
            std::copy(key_slot.begin(), key_slot.end(), key_data.begin() + static_cast<ptrdiff_t>(b * key_per_batch));
            std::copy(value_slot.begin(), value_slot.end(), value_data.begin() + static_cast<ptrdiff_t>(b * value_per_batch));
        }
        assign_precision(key, key_data);
        assign_precision(value, value_data);
        tensors[1] = key;
        tensors[2] = value;
    }

    // Synthetic selection: the up-to-m_shape.max_blocks *complete* blocks immediately strictly before the
    // query's own block, in ascending order -- per the documented producer contract, the current
    // (possibly incomplete) block must never appear in sel_indices/sel_count, and sel_indices must
    // be strictly ascending and duplicate-free; the consumer derives and attends the current block
    // automatically instead. Picking a run of consecutive preceding blocks (rather than just one)
    // exercises the CPU node's AMX adjacent-block-merging path. Shared across batch items (only the
    // underlying K/V differs), so a bug that mixed up batch indices for the K/V cache still gets
    // caught via numerically different output.
    // Block ids are counted from the start of each batch item's *real* (non-padded) history --
    // physical row `pad + block_id * sel_block_size`, per the producer contract -- so with padding
    // the query's own logical block is (l - pad) / sel_block_size rather than l / sel_block_size.
    // pad == 0 for every batch when m_key_length is empty, reducing to the original formula exactly.
    std::vector<int32_t> sel_indices_flat(m_batch * m_h_sel * m_shape.seq_len * m_shape.max_blocks, -1);
    std::vector<int32_t> sel_count_flat(m_batch * m_h_sel * m_shape.seq_len, 0);
    for (size_t b = 0; b < m_batch; b++) {
        const int64_t pad =
            m_key_length.empty() ? 0 : static_cast<int64_t>(m_shape.seq_len) - static_cast<int64_t>(m_key_length[b]);
        for (size_t l = 0; l < m_shape.seq_len; l++) {
            const int64_t rel_pos = static_cast<int64_t>(l) - pad;
            const int32_t cur_block =
                rel_pos >= 0 ? static_cast<int32_t>(rel_pos / static_cast<int64_t>(m_sel_block_size)) : 0;
            const size_t num_sel =
                rel_pos >= 0 ? std::min(static_cast<size_t>(cur_block) / m_shape.block_step, m_shape.max_blocks) : 0;
            for (size_t sh = 0; sh < m_h_sel; sh++) {
                const size_t base = (b * m_h_sel + sh) * m_shape.seq_len + l;
                for (size_t k = 0; k < num_sel; k++) {
                    sel_indices_flat[base * m_shape.max_blocks + k] =
                        cur_block - static_cast<int32_t>(m_shape.block_step * (num_sel - k));
                }
                sel_count_flat[base] = static_cast<int32_t>(num_sel);
            }
        }
    }
    tensors[3] = make_index(sel_indices_flat, ov::Shape{m_batch, m_h_sel, m_shape.seq_len, m_shape.max_blocks});
    tensors[4] = make_index(sel_count_flat, ov::Shape{m_batch, m_h_sel, m_shape.seq_len});

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
    auto sel_indices = to_i32(params[3]);
    auto sel_count = to_i32(params[4]);

    auto output = run_sparse_sdpa_reference(m_shape.seq_len,
                                            m_shape.max_blocks,
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

    ov::Tensor output_tensor(ov::element::f32, ov::Shape{m_batch, kHq, m_shape.seq_len, kDv});
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
