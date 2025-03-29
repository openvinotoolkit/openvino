// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using roialignrotatedParams = std::tuple<std::vector<InputShape>,  // Feature map shape
                                         int,                      // Num of Rois
                                         int,                      // Pooled h
                                         int,                      // Pooled w
                                         int,                      // Sampling ratio
                                         float,                    // Spatial scale
                                         bool,                     // Clockwise mode
                                         ov::element::Type,        // Model type
                                         ov::test::TargetDevice>;  // Device name

class ROIAlignRotatedLayerTest : public testing::WithParamInterface<roialignrotatedParams>,
                                 virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<roialignrotatedParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
