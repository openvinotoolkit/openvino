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
using PowerParamsTuple = typename std::tuple<
    std::vector<InputShape>, // Input shapes
    ov::element::Type,       // Model type
    std::vector<float>,      // Power
    std::string>;            // Device name

class PowerLayerTest:
        public testing::WithParamInterface<PowerParamsTuple>,
        virtual public ov::test::SubgraphBaseTest{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PowerParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
