// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
using ov::test::utils::MinMaxOpType;

static std::map<MinMaxOpType, std::string> extremumNames = {
        {MinMaxOpType::MINIMUM, "MINIMUM"},
        {MinMaxOpType::MAXIMUM, "MAXIMUM"}
};

using MaxMinParamsTuple = typename std::tuple<
        std::vector<InputShape>,          // Input shapes
        ov::test::utils::MinMaxOpType,    // Operation type
        ov::element::Type,                // Model type
        ov::test::utils::InputLayerType,  // Secondary input type
        std::string>;                     // Device name

class MaxMinLayerTest:
        public testing::WithParamInterface<MaxMinParamsTuple>,
        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MaxMinParamsTuple>& obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
