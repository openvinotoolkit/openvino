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
typedef std::tuple<
        double,                        // Alpha
        double,                        // Beta
        double,                        // Bias
        size_t,                        // Size
        std::vector<int64_t>,          // Reduction axes
        ov::element::Type,             // Network precision
        std::vector<InputShape>,      // Input shapes
        std::string                    // Device name
> lrnLayerTestParamsSet;

class LrnLayerTest
        : public testing::WithParamInterface<lrnLayerTestParamsSet>,
          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<lrnLayerTestParamsSet>& obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
