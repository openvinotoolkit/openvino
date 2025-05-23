// Copyright (C) 2023 Intel Corporation
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

using ActivationLayerCPUTestParamSet =
    std::tuple<std::vector<ov::test::InputShape>,                      // Input shapes
               std::vector<size_t>,                                    // Activation shapes
               std::pair<utils::ActivationTypes, std::vector<float>>,  // Activation type and constant value
               ov::element::Type,                                      // Net precision
               ov::element::Type,                                      // Input precision
               ov::element::Type,                                      // Output precision
               CPUTestUtils::CPUSpecificParams,
               bool>;

class ActivationLayerCPUTest : public testing::WithParamInterface<ActivationLayerCPUTestParamSet>,
                               virtual public ov::test::SubgraphBaseTest,
                               public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;

private:
    ov::element::Type netPrecision = ov::element::dynamic;
    utils::ActivationTypes activationType = utils::ActivationTypes::None;

    std::string getPrimitiveType(
        const utils::ActivationTypes& activation_type,
        const ov::element::Type_t& element_type,
        const std::vector<std::pair<ov::PartialShape, std::vector<ov::Shape>>>& input_shapes) const;
};

namespace Activation {

const std::vector<size_t> activationShapes();

const std::map<utils::ActivationTypes, std::vector<std::vector<float>>>& activationTypes();
const std::map<utils::ActivationTypes, std::vector<std::vector<float>>>& activationTypesSnippets();

const std::vector<ov::element::Type>& netPrc();

/* ============= Activation (1D) ============= */
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams3D();

const std::vector<std::vector<ov::Shape>>& basic3D();

/* ============= Activation (2D) ============= */
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams4D();

const std::vector<std::vector<ov::Shape>>& basic4D();

/* ============= Activation (3D) ============= */
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams5D();

const std::vector<std::vector<ov::Shape>>& basic5D();

const std::map<utils::ActivationTypes, std::vector<std::vector<float>>>& activationTypesDynamicMath();

const std::vector<ov::element::Type>& netPrecisions();

const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParamsDynamicMath();

const std::vector<std::vector<ov::test::InputShape>>& dynamicMathBasic();

}  // namespace Activation
}  // namespace test
}  // namespace ov
