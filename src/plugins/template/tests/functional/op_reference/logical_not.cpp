// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/logical_not.hpp"
#include "logical.hpp"

using namespace ov;
using LogicalTypes = ngraph::helpers::LogicalTypes;

namespace reference_tests {
namespace LogicalOpsRefTestDefinitions {
namespace {

std::vector<RefLogicalParams> generateLogicalParams() {
    std::vector<RefLogicalParams> logicalParams {
        Builder {}
            .opType(LogicalTypes::LOGICAL_NOT)
            .inputs({{{2, 2}, element::boolean, std::vector<char> {true, false, true, false}}})
            .expected({{2, 2}, element::boolean, std::vector<char> {false, true, false, true}})};
    return logicalParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_LogicalNot_With_Hardcoded_Refs, ReferenceLogicalLayerTest, ::testing::ValuesIn(generateLogicalParams()),
                         ReferenceLogicalLayerTest::getTestCaseName);

}  // namespace
}  // namespace LogicalOpsRefTestDefinitions
}  // namespace reference_tests
