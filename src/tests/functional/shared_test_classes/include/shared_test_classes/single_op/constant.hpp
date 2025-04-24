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
using constantParamsTuple = typename std::tuple<
    ov::Shape,                   // Constant data shape
    ov::element::Type,           // Constant data precision
    std::vector<std::string>,    // Constant elements
    std::string>;                // Device name

class ConstantLayerTest : public testing::WithParamInterface<constantParamsTuple>,
                          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<constantParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
