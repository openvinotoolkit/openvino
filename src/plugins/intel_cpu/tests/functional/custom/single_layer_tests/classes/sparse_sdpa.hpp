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

// target device, data precision (query/key/value), h_sel, causal, batch (B; plain batched dense
// K/V history, see classes/sparse_sdpa.cpp), k_head_size, sel_block_size, per-batch key_length
// (empty = no padding / 5-input op, matching today's behavior exactly; otherwise one entry per
// batch item in [1, kCacheLen], builds the 6-input op with KEY_LENGTH)
using SparseSDPALayerTestParams =
    std::tuple<std::string, ov::element::Type, size_t, bool, size_t, size_t, size_t, std::vector<size_t>>;

class SparseSDPALayerCPUTest : public testing::WithParamInterface<SparseSDPALayerTestParams>, virtual public SubgraphBaseTest {
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
    size_t m_h_sel = 1;
    bool m_causal = true;
    size_t m_batch = 1;
    size_t m_k_head_size = 16;
    size_t m_sel_block_size = 2;
    float m_scale = 0.25F;
    std::vector<size_t> m_key_length;
};

}  // namespace test
}  // namespace ov
