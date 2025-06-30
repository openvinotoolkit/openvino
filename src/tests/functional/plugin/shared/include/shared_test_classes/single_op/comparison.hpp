// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <map>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

typedef std::tuple<
    std::vector<InputShape>,             // Input shapes tuple
    ov::test::utils::ComparisonTypes,    // Comparison op type
    ov::test::utils::InputLayerType,     // Second input type
    ov::element::Type,                   // Model type
    std::string,                         // Device name
    std::map<std::string, std::string>   // Additional network configuration
> ComparisonTestParams;

class ComparisonLayerTest : public testing::WithParamInterface<ComparisonTestParams>,
    virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ComparisonTestParams> &obj);
};
} // namespace test
} // namespace ov
