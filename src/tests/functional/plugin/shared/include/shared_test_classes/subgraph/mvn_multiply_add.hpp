// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<std::pair<ov::Shape, ov::Shape>,  // Input shape, Constant shape
                   ov::element::Type,                // Data precision
                   ov::element::Type,                // Axes precision
                   std::vector<int>,                 // Axes
                   bool,                             // Normalize variance
                   float,                            // Epsilon
                   std::string,                      // Epsilon mode
                   std::string                       // Device name
                   >
    mvnMultiplyAddParams;

class MVNMultiplyAdd : public testing::WithParamInterface<mvnMultiplyAddParams>,
                       virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mvnMultiplyAddParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
