// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

using ComparisonLayerCPUTestParamSet =
    std::tuple<std::vector<ov::test::InputShape>,  // Input shapes
               utils::ComparisonTypes,             // Comparison type
               ov::test::utils::InputLayerType,    // Second input type
               ov::element::Type,                  // Model precision
               ov::element::Type,                  // Infer precision
               bool>;                              // Enforce Snippets

class ComparisonLayerCPUTest : public testing::WithParamInterface<ComparisonLayerCPUTestParamSet>,
                               virtual public ov::test::SubgraphBaseTest,
                               public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ComparisonLayerCPUTestParamSet> &obj);

protected:
    void SetUp() override;

private:
    std::string getPrimitiveType(const utils::ComparisonTypes& comparison_type) const;
};


namespace comparison {

const std::vector<std::vector<InputShape>>& inShapesWithParameter();
const std::vector<std::vector<InputShape>>& inShapesWithConstant();

const std::vector<utils::ComparisonTypes>& comparisonTypes();
const std::vector<utils::ComparisonTypes>& comparisonTypesSnippets();

const std::vector<ov::element::Type>& modelPrc();
const std::vector<ov::element::Type> inferPrc();

}  // namespace comparison
}  // namespace test
}  // namespace ov
