// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

using PagedGatedDeltaNetLayerParams = std::tuple<int32_t,
                                                 int32_t,
                                                 int32_t,
                                                 int32_t,
                                                 std::vector<int32_t>,
                                                 std::vector<int32_t>,
                                                 ov::element::Type,
                                                 std::string>;  // qk_heads, v_heads, qk_head_size, v_head_size,
                                                                // seq_lengths, cache_intervals, element_type,
                                                                // target_device

class PagedGatedDeltaNetLayerTest : public testing::WithParamInterface<PagedGatedDeltaNetLayerParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedGatedDeltaNetLayerParams>& obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void SetUp() override;

private:
    std::vector<float> ref_recurrent_state_table;
};

}  // namespace ov::test
