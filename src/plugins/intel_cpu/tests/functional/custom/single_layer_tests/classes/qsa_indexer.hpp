// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// Host-vector helpers of the reference below (float <-> f32/f16/bf16 tensors).
std::vector<float> read_tensor_to_f32(const ov::Tensor& t);
void write_f32_to_tensor(ov::Tensor& t, const std::vector<float>& data);
ov::Tensor make_zero_tensor(const ov::element::Type& precision, const ov::Shape& shape);

// Independent (host-vector based) reference of one QSAIndexer call, re-implementing
// new_spec/qsa-prepare-17.rst's padding-aware prepare+selection pseudocode directly -- NOT a copy
// of the CPU node's PlainTensor-based indexing/control flow. `summary_cache`/`past_raw_k`/
// `past_rope` are caller-owned state threaded in by reference across sequential calls, mirroring
// the production op's state. Also used by the QSAPrepare + SparseAttentionIndexer fusion subgraph test.
struct QSAIndexerOracle {
    size_t h_iq = 0;
    size_t h_ik = 0;
    size_t h_sel = 0;
    size_t d_idx = 0;
    size_t rotary_dim = 0;
    int64_t r = 0;
    size_t block_topk = 0;
    float eps = 1e-6F;
    float scale = 0.25F;
    size_t summary_cap = 0;

    std::vector<float> summary_cache;  // [B, h_ik, summary_cap, d_idx]
    std::vector<float> past_raw_k;     // [B, h_ik, r - 1, d_idx]
    std::vector<float> past_rope;      // [B, 2 * rotary_dim]

    struct Output {
        std::vector<int32_t> sel_indices;
        std::vector<int32_t> sel_count;
    };

    Output run_call(size_t batch,
                    int64_t length,
                    const std::vector<int32_t>& key_length,
                    const std::vector<float>& q_idx,
                    const std::vector<float>& k_idx,
                    const std::vector<float>& rope_cos,
                    const std::vector<float>& rope_sin,
                    const std::vector<float>& q_norm_weight,
                    const std::vector<float>& k_norm_weight);
};

// Which fixed per-call (L, key_length) schedule SetUp() builds. kDensePrefill is a single call
// with key_length[b] == L for every row (no past, no padding) -- basic correctness coverage.
// kWorkedExample reproduces new_spec/qsa-prepare-17.rst's worked example verbatim: compress_ratio
// 4, a prefill call (L=10, key_length=[10,7], row 1 left-padded by 3) followed by a decode call
// (L=1, key_length=[11,8]) that completes row 1's second block. Forces batch to 2.
enum class QSAIndexerScenario { kDensePrefill, kWorkedExample };

// target device, data precision (q_idx/k_idx/rotary/caches/norm weights),
// H_iq (index-query head count), H_ik (index-key head count), h_sel (selection-head count),
// batch (ignored for kWorkedExample, which forces batch == 2), scenario
using QSAIndexerLayerTestParams =
    std::tuple<std::string, ov::element::Type, size_t, size_t, size_t, size_t, QSAIndexerScenario>;

class QSAIndexerLayerCPUTest : public testing::WithParamInterface<QSAIndexerLayerTestParams>,
                               virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QSAIndexerLayerTestParams>& obj);

protected:
    void SetUp() override;
    void run() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

private:
    struct CallSpec {
        int64_t length = 0;
        std::vector<int32_t> key_length;
    };

    std::vector<ov::Tensor> run_call(const CallSpec& call, size_t call_idx);

    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_h_iq = 2;
    size_t m_h_ik = 2;
    size_t m_h_sel = 1;
    size_t m_batch = 1;
    QSAIndexerScenario m_scenario = QSAIndexerScenario::kDensePrefill;

    size_t m_compress_ratio = 2;
    size_t m_block_topk = 2;
    size_t m_rotary_dim = 8;
    size_t m_didx = 16;
    size_t m_summary_cap = 0;  // block capacity of the host oracle's summary cache only
    std::vector<CallSpec> m_calls;

    ov::InferRequest m_infer_request;
    ov::Tensor m_summary_cache_tensor;
    ov::Tensor m_past_raw_k_tensor;
    ov::Tensor m_past_rope_tensor;
    ov::Tensor m_q_norm_weight_tensor;
    ov::Tensor m_k_norm_weight_tensor;

    // Independent (host-vector based) oracle state, persisted across the calls in m_calls --
    // mirrors the production op's three state tensors (fed back from its state outputs), but is never read from the
    // compiled model's own tensors.
    std::vector<float> m_oracle_summary_cache;
    std::vector<float> m_oracle_past_raw_k;
    std::vector<float> m_oracle_past_rope;
};

}  // namespace test
}  // namespace ov
