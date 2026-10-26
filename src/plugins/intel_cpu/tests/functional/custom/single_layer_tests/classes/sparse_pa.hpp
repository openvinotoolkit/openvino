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

// target device, data precision (query/key/value/caches), num_selection_heads, causal,
// num_sequences (each sequence gets its own past_len and its own run of physical cache blocks,
// see classes/sparse_pa.cpp), k_head_size, key_block_size, page_size (KEY_CACHE/VALUE_CACHE dim 2;
// must be a multiple of key_block_size), with_qb_start (false: the optional QB_START input is omitted, i.e.
// subsequence_begins is used)
using SparsePALayerTestParams =
    std::tuple<std::string, ov::element::Type, size_t, bool, size_t, size_t, size_t, size_t, bool>;

class SparsePALayerCPUTest : public testing::WithParamInterface<SparsePALayerTestParams>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparsePALayerTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

private:
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_num_selection_heads = 1;
    bool m_causal = true;
    size_t m_num_sequences = 1;
    size_t m_k_head_size = 16;
    size_t m_key_block_size = 2;
    size_t m_page_size = 4;
    size_t m_seq_blocks = 4;
    float m_scale = 0.25F;
    bool m_with_qb_start = true;
};

}  // namespace test
}  // namespace ov