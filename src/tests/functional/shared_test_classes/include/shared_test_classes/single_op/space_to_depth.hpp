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
using spaceToDepthParamsTuple = typename std::tuple<
        std::vector<InputShape>,                        // Input shape
        ov::element::Type,                              // Model type
        ov::op::v0::SpaceToDepth::SpaceToDepthMode,     // Mode
        std::size_t,                                    // Block size
        std::string>;                                   // Device name

class SpaceToDepthLayerTest : public testing::WithParamInterface<spaceToDepthParamsTuple>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<spaceToDepthParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
