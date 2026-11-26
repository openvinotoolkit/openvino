// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

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
               bool,                               // Enforce Snippets
               std::optional<ov::test::utils::InputGenerateData>>;  // Value window for i32 inputs

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

// i32 values that are not exactly representable in f32: around +-2^24 and near the i32 limits. Each window holds 3
// consecutive values, so both inputs are often equal or differ by 1. Inline so per-test targets can reuse it.
inline const std::vector<std::optional<ov::test::utils::InputGenerateData>>& i32Windows() {
    static const std::vector<std::optional<ov::test::utils::InputGenerateData>> windows{
        ov::test::utils::InputGenerateData((1 << 24) - 1, 3),
        ov::test::utils::InputGenerateData(-(1 << 24) - 1, 3),
        ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::max() - 2, 3),
        ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::min(), 3),
    };

    return windows;
}

const std::vector<ov::element::Type>& modelPrc();
const std::vector<ov::element::Type> inferPrc();

}  // namespace comparison

}  // namespace test
}  // namespace ov
