// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using FakeConvertParams = std::tuple<std::vector<InputShape>,  // Data shape
                                     Shape,                    // Scale shape
                                     Shape,                    // Shift shape
                                     ov::element::Type,        // Input precision
                                     ov::element::Type,        // Ddestination precision
                                     bool,                     // Default shift
                                     std::string>;             // Device name

class FakeConvertLayerTest : public testing::WithParamInterface<FakeConvertParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FakeConvertParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
