// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// A prefill over dense K/V (S == L). block_step: distance between selected blocks; shuffle: random order. key_length:
// empty for the 5-input op, else one entry per batch item in [1, seq_len].
using SparseSDPALayerTestParams = std::tuple<size_t,               // batch
                                             size_t,               // num_q_heads (Hq)
                                             size_t,               // num_kv_heads (Hkv), divides Hq
                                             size_t,               // k_head_size
                                             size_t,               // v_head_size
                                             size_t,               // seq_len (L == S)
                                             size_t,               // num_selection_heads (h_sel)
                                             size_t,               // sel_block_size
                                             size_t,               // max_blocks (K_max, width of block_indices)
                                             size_t,               // block_step
                                             bool,                 // causal
                                             bool,                 // shuffle
                                             std::vector<size_t>,  // key_length
                                             ov::element::Type,    // data_precision (query/key/value)
                                             std::string>;         // device

class SparseSDPALayerCPUTest : public testing::WithParamInterface<SparseSDPALayerTestParams>,
                               virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparseSDPALayerTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

private:
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_batch = 1;
    size_t m_num_q_heads = 4;
    size_t m_num_kv_heads = 2;
    size_t m_k_head_size = 16;
    size_t m_v_head_size = 16;
    size_t m_seq_len = 40;  // taken from the target shape of the query
    size_t m_h_sel = 1;
    size_t m_sel_block_size = 2;
    size_t m_max_blocks = 10;
    size_t m_block_step = 1;
    bool m_causal = true;
    bool m_shuffle = false;
    float m_scale = 0.25F;
    std::vector<size_t> m_key_length;
};

}  // namespace test
}  // namespace ov
