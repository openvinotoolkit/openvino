// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "logical.hpp"

using namespace ngraph;
using namespace InferenceEngine;
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
