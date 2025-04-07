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
using regionYoloParamsTuple = std::tuple<
    std::vector<size_t>,            // Input shape
    size_t,                         // Classes
    size_t,                         // Coordinates
    size_t,                         // Num regions
    bool,                           // Do softmax
    std::vector<int64_t>,           // Mask
    int,                            // Start axis
    int,                            // End axis
    ov::element::Type,              // Model type
    ov::test::TargetDevice          // Device name
>;

class RegionYoloLayerTest : public testing::WithParamInterface<regionYoloParamsTuple>,
                            virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<regionYoloParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
