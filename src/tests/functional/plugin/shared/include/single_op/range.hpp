// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using RangeParams = std::tuple<
    float,                          // Start
    float,                          // Stop
    float,                          // Step
    ov::element::Type,              // Model type
    ov::test::TargetDevice          // Device name
>;

class RangeLayerTest : public testing::WithParamInterface<RangeParams>,
                       virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RangeParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
