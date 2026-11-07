// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

enum class ComparisonGraphType { COMPARE, COMPARE_LOGICAL_NOT, ADD_COMPARE };

using ComparisonLayerCPUTestParamSet = std::tuple<std::vector<ov::test::InputShape>,  // Input shapes
                                                  utils::ComparisonTypes,             // Comparison type
                                                  ov::test::utils::InputLayerType,    // Second input type
                                                  ov::element::Type,                  // Model precision
                                                  ov::element::Type,                  // Infer precision
                                                  bool,                               // Enforce Snippets
                                                  bool,                               // Use large i32 values
                                                  ComparisonGraphType>;               // Graph pattern

class ComparisonLayerCPUTest : public testing::WithParamInterface<ComparisonLayerCPUTestParamSet>,
                               virtual public ov::test::SubgraphBaseTest,
                               public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ComparisonLayerCPUTestParamSet> &obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

private:
    std::string getPrimitiveType(const utils::ComparisonTypes& comparison_type, ov::element::Type modelType) const;

    std::optional<ov::test::utils::InputGenerateData> i32Window;
};

namespace comparison {

const std::vector<std::vector<InputShape>>& inShapesWithParameter();
const std::vector<std::vector<InputShape>>& inShapesWithConstant();

const std::vector<utils::ComparisonTypes>& comparisonTypes();
const std::vector<bool>& enforceSnippets();

const std::vector<ov::element::Type>& modelPrc();
const std::vector<ov::element::Type> inferPrc();

}  // namespace comparison

}  // namespace test
}  // namespace ov
