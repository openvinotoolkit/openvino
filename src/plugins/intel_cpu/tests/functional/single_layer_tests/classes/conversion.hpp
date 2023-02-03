// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/activation.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"

namespace CPULayerTestsDefinitions  {
using convertLayerTestParamsSet = std::tuple<
        ov::test::InputShape,        // input shapes
        ov::test::ElementType,       // input precision
        ov::test::ElementType,       // output precision
        ov::AnyMap,                  // Additional plugin configuration
        CPUTestUtils::CPUSpecificParams
>;

class ConvertCPULayerTest : public testing::WithParamInterface<convertLayerTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj);
    static bool isInOutPrecisionSupported(ov::test::ElementType inPrc, ov::test::ElementType outPrc);
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

private:
    ov::test::ElementType inPrc, outPrc;
};

namespace Conversion {
    const std::vector<ov::test::InputShape>& inShapes_4D_static();
    const std::vector<ov::test::InputShape>& inShapes_4D_dynamic();
    const std::vector<ov::test::ElementType>& precisions();

} // namespace Conversion
} // namespace CPULayerTestsDefinitions
