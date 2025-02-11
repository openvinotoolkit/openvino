// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {
enum SpecialValue { none, nan, inf, overflow };

using convertLayerTestParamsSet = std::tuple<InputShape,         // input shapes
                                             ov::element::Type,  // input precision
                                             ov::element::Type,  // output precision
                                             SpecialValue,       // Specail value
                                             CPUSpecificParams>;

class ConvertCPULayerTest : public testing::WithParamInterface<convertLayerTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj);
    static bool isInOutPrecisionSupported(ov::element::Type inPrc, ov::element::Type outPrc);
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void validate() override;
    virtual void validate_out_prc() const;

    ov::element::Type inPrc, outPrc;
private:
    ov::test::SpecialValue special_value;
};

class ConvertToBooleanCPULayerTest : public ConvertCPULayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void validate_out_prc() const override;
};

namespace Conversion {
const std::vector<InputShape>& inShapes_4D_static();
const std::vector<InputShape>& inShapes_4D_dynamic();
const std::vector<InputShape>& inShapes_7D_static();
const std::vector<InputShape>& inShapes_7D_dynamic();
const std::vector<ov::element::Type>& precisions();
}  // namespace Conversion
}  // namespace test
}  // namespace ov