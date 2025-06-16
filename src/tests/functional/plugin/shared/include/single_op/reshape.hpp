// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using reshapeParams = std::tuple<
    bool,                           // SpecialZero
    ov::element::Type,              // Model type
    std::vector<size_t>,            // Input shapes
    std::vector<int64_t>,           // OutForm shapes
    ov::test::TargetDevice          // Device name
>;
class ReshapeLayerTest : public testing::WithParamInterface<reshapeParams>,
                         virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<reshapeParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
