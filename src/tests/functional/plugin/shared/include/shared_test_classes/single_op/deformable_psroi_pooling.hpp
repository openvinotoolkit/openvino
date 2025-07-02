// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using deformablePSROISpecificParams = std::tuple<
    int64_t,                        // output_dim
    int64_t,                        // group_size
    float,                          // spatial_scale
    std::vector<int64_t>,           // spatial_bins_x_y
    float,                          // trans_std
    int64_t>;                       // part_size

using deformablePSROILayerTestParams = std::tuple<
    deformablePSROISpecificParams,
    std::vector<InputShape>,    // data input shape
    ov::element::Type,          // Net type
    std::string>;               // Device name

class DeformablePSROIPoolingLayerTest : public testing::WithParamInterface<deformablePSROILayerTestParams>,
    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<deformablePSROILayerTestParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
