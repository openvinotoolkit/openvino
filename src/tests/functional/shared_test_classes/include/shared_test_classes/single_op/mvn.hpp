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
        std::vector<InputShape>,  // Input shapes
        ov::element::Type,        // Model type
        ov::AxisSet,              // Reduction axes
        bool,                     // Across channels
        bool,                     // Normalize variance
        double,                   // Epsilon
        std::string               // Device name
    > mvn1Params;

class Mvn1LayerTest : public testing::WithParamInterface<mvn1Params>,
                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mvn1Params>& obj);

protected:
    void SetUp() override;
};

typedef std::tuple<
        std::vector<InputShape>,  // Input shapes
        ov::element::Type,        // Model type
        ov::element::Type,        // Axes type
        std::vector<int>,         // Axes
        bool,                     // Normalize variance
        float,                    // Epsilon
        std::string,              // Epsilon mode
        std::string               // Device name
    > mvn6Params;

class Mvn6LayerTest : public testing::WithParamInterface<mvn6Params>,
                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mvn6Params>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
