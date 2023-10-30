// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using GridSampleParams = std::tuple<ov::Shape,                                  // Data shape
                                    ov::Shape,                                  // Grid shape
                                    bool,                                       // Align corners
                                    ov::op::v9::GridSample::InterpolationMode,  // Mode
                                    ov::op::v9::GridSample::PaddingMode,        // Padding mode
                                    ov::element::Type,                          // Data precision
                                    ov::element::Type,                          // Grid precision
                                    std::string>;                               // Device name

class GridSampleLayerTest : public testing::WithParamInterface<GridSampleParams>,
                            virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GridSampleParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
