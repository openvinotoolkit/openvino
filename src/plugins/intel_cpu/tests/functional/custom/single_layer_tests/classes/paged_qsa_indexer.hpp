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
// see classes/paged_qsa_indexer.cpp)
using PagedQSAIndexerLayerTestParams = std::tuple<std::string, ov::element::Type, size_t, size_t, size_t, size_t>;

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
};

}  // namespace test
}  // namespace ov