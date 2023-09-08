// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <map>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace ov {
namespace test {

typedef std::tuple<
    std::vector<InputShape>,             // Input shapes tuple
    ngraph::helpers::ComparisonTypes,    // Comparison op type
    ngraph::helpers::InputLayerType,     // Second input type
    ov::element::Type,                   // In type
    std::string,                         // Device name
    std::map<std::string, std::string>   // Additional network configuration
> ComparisonTestParams;

class ComparisonLayerTest : public testing::WithParamInterface<ComparisonTestParams>,
    virtual public ov::test::SubgraphBaseTest {
    ngraph::helpers::ComparisonTypes comparison_op_type;
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ComparisonTestParams> &obj);
};
} // namespace test
} // namespace ov
