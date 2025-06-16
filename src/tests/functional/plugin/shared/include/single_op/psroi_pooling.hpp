// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>


#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

using psroiParams = std::tuple<
    std::vector<size_t>,            // Input shape
    std::vector<size_t>,            // Coords shape
    size_t,                         // Output_dim
    size_t,                         // group_size
    float,                          // Spatial scale
    size_t,                         // spatial_bins_x
    size_t,                         // spatial_bins_y
    std::string,                    // Mode
    ov::element::Type,              // Model type
    ov::test::TargetDevice          // Device name
>;

class PSROIPoolingLayerTest : public testing::WithParamInterface<psroiParams>,
                              virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<psroiParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
