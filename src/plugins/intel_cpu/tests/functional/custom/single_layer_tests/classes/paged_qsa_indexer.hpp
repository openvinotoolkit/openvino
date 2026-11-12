// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "openvino/core/type/element_type.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// target device, data precision (q_idx/k_idx/rotary/caches/norm weights),
// H_iq (index-query head count), H_ik (index-key head count), h_sel (selection-head count),
// num_sequences (each sequence gets its own past_len and its own run of physical cache blocks,
// see classes/paged_qsa_indexer.cpp), new_tokens (the tokens every sequence adds in the call: 1 is a decode step),
// page_size (tokens of a cache page: a multiple of the compress ratio; 32 blocks of the scoring can lie in one page or
// run over several)
using PagedQSAIndexerLayerTestParams =
    std::tuple<std::string, ov::element::Type, size_t, size_t, size_t, size_t, size_t, size_t>;

class PagedQSAIndexerLayerCPUTest : public testing::WithParamInterface<PagedQSAIndexerLayerTestParams>,
                                    virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedQSAIndexerLayerTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

private:
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_h_iq = 2;
    size_t m_h_ik = 2;
    size_t m_h_sel = 1;
    size_t m_num_sequences = 1;
    size_t m_l = 7;  // new tokens of every sequence
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