// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

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
    size_t m_summary_cap = 0;
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
