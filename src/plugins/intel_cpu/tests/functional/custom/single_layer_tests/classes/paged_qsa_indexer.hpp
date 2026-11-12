// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// target device, data precision (q_idx/k_idx/rotary/caches/norm weights),
// H_iq (index-query head count), H_ik (index-key head count), h_sel (selection-head count),
// num_sequences (each sequence gets its own past_len and its own run of physical cache blocks,
// see classes/paged_qsa_indexer.cpp), new_tokens (an InputShape: the dynamic shape is {-1} and every target shape {n}
// is one call that adds n tokens to every sequence. The first shape is the prefill, the following ones are decode steps
// (n == 1) or chunks, and the calls continue the same sequences and caches, so the number of calls is the number of
// target shapes),
// page_size (tokens of a cache page: a multiple of the compress ratio; 32 blocks of the scoring can lie in one page or
// run over several)
using PagedQSAIndexerLayerTestParams =
    std::tuple<std::string, ov::element::Type, size_t, size_t, size_t, size_t, InputShape, size_t>;

class PagedQSAIndexerLayerCPUTest : public testing::WithParamInterface<PagedQSAIndexerLayerTestParams>,
                                    virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedQSAIndexerLayerTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void validate() override;

private:
    // The past length of a sequence in the current call: the calls of a test continue the sequences.
    size_t sequence_past_len(size_t seq) const;
    void check_caches_updated() const;

    // The tokens of a sequence over the calls so far, which the oracle replays (the result of a call depends on the
    // caches the previous calls filled).
    struct SequenceHistory {
        std::vector<float> q_idx, k_idx, rope_cos, rope_sin;
    };

    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    std::vector<ov::Tensor> m_caches;  // raw key, block rope and block summary caches, kept across the calls
    std::vector<size_t> m_calls;       // new tokens of every sequence in each call
    std::vector<size_t> m_past;        // past length of every sequence in the current call
    std::vector<SequenceHistory> m_history;
    size_t m_next_call = 0;
    size_t m_call = 0;   // index of the current call
    size_t m_cur_l = 0;  // new tokens of every sequence in the current call
    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_h_iq = 2;
    size_t m_h_ik = 2;
    size_t m_h_sel = 1;
    size_t m_num_sequences = 1;
    // Share of the elements of an output that may differ: selections of (almost) tied blocks flip with the rounding of
    // a 16-bit cache, which the f32 oracle does not do. 0 for f32; for the small tests of the 16-bit types the share
    // of a few elements rounds down to none, so they stay exact.
    double m_mismatch_share = 0.0;
    size_t m_page_size = 4;
    size_t m_slots_per_page = 2;  // blocks of a page: page_size / compress_ratio
    size_t m_seq_blocks = 4;      // physical cache blocks of every sequence, enough for its tokens and a spare one
};

}  // namespace test
}  // namespace ov