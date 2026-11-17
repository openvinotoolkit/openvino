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

// Host-vector reference of one QSAIndexer call, written from new_spec/qsa-prepare-17.rst, not from the node's
// code. `summary_cache`, `past_raw_k` and `past_rope` are state the caller threads across calls.
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
    // chosen blocks by descending score (ties: lower id first), not by ascending id
    bool order_by_score = false;
    // the steps of the query that the op applies (the op leaves the others to the graph in front of it)
    bool query_norm = true;
    bool query_rope = true;

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

// The fixed per-call (L, key_length) schedule SetUp() builds.
enum class QSAIndexerScenario {
    // one call, key_length[b] == L, no past, no padding
    kDensePrefill,
    // the worked example of new_spec/qsa-prepare-17.rst: left-padded prefill, then a decode call (forces batch 2)
    kWorkedExample,
    // a prefill where row 1 is all padding, then a call where that row starts from scratch (forces batch 2)
    kFullyPaddedRow,
    // three calls with L > 1 and different past lengths per row; blocks complete mid-call and at a call's end
    kRaggedCalls,
    // The scenarios below have no reference: they check only the selection invariants (distinct, visible, ordered
    // block ids). One 80-token prefill, compress_ratio 2, so the vectorized selection runs.
    // a NaN in the keys of a few tokens
    kNaNScores,
    // query norm weights that overflow the scores to +-Inf and NaN
    kOverflowScores,
    // query norm weights that make a row's scores differ by less than 256 / FLT_MAX
    kTinyScores
};

// The steps of the query that the op applies: the query_norm / query_rope attributes.
enum class QSAIndexerQueryFusionType { kNormAndRope, kNormOnly, kRopeOnly, kNone };

// target device, data precision, H_iq, H_ik, h_sel, batch (only kDensePrefill uses it; the other scenarios force
// 2), scenario, order_by_score, query steps
using QSAIndexerLayerTestParams = std::tuple<std::string,
                                             ov::element::Type,
                                             size_t,
                                             size_t,
                                             size_t,
                                             size_t,
                                             QSAIndexerScenario,
                                             bool,
                                             QSAIndexerQueryFusionType>;

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
    bool is_degenerate_scores_scenario() const;
    void check_selection_invariants(const CallSpec& call, const std::vector<ov::Tensor>& outs) const;

    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_h_iq = 2;
    size_t m_h_ik = 2;
    size_t m_h_sel = 1;
    size_t m_batch = 1;
    bool m_order_by_score = false;
    bool m_query_norm = true;
    bool m_query_rope = true;
    // Share of output elements that may differ: 16-bit types can flip the score order of near-equal blocks.
    double m_mismatch_share = 0.0;
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

    // Oracle copy of the op's three state tensors, kept across m_calls and never read from the compiled model.
    std::vector<float> m_oracle_summary_cache;
    std::vector<float> m_oracle_past_raw_k;
    std::vector<float> m_oracle_past_rope;
};

}  // namespace test
}  // namespace ov
