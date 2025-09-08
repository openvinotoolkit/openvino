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
typedef std::tuple<
        size_t,                         // Num splits
        int64_t,                        // Axis
        ov::element::Type,              // Model type
        std::vector<InputShape>,        // Input shapes
        std::vector<size_t>,            // Used outputs indices
        std::string                     // Target device name
> splitParams;

class SplitLayerTest : public testing::WithParamInterface<splitParams>,
                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<splitParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
