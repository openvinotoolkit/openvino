// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_models/builders.hpp>
#include "ie_common.h"
#include "ov_models/utils/ov_helpers.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

class InputNoReorderEltwiseBF16 : virtual public LayerTestsUtils::LayerTestsCommon,
                                  public CPUTestsBase {
protected:
    void SetUp() override {
        auto netPrecision = inPrc = Precision::FP32;
        outPrc = Precision::BF16;
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::map<std::string, std::string> additional_config{{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}};
        configuration.insert(additional_config.begin(), additional_config.end());

        std::vector<size_t> inputShape {2, 4, 4, 1};
        auto eltwiseType = ngraph::helpers::EltwiseTypes::ADD;
        auto secondaryInputType = ngraph::helpers::InputLayerType::CONSTANT;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        std::shared_ptr<ngraph::Node> secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, inputShape);
        auto eltwise = ngraph::builder::makeEltwise(input[0], secondaryInput, eltwiseType);

        function = makeNgraphFunction(ngPrc, input, eltwise, "Eltwise");
    }
};

/* FP32 network with enforced BF16 precision.
 * Test that no Reorder (or Convert) is inserted after Input.
 * Eltwise performs the conversion by itself.

    Input[FP32]        Constant[FP32]
          \                 /
           \               /
            X  No Reorder X
             \           /
           Eltwise[FP32->BF16] (or Subgraph[FP32->BF16])
                  |
                  |
             Output[BF16]
*/
TEST_F(InputNoReorderEltwiseBF16, smoke_CompareWithRefs) {
    Run();

    CheckNumberOfNodesWithType(executableNetwork, "Reorder", 0);
    CheckNumberOfNodesWithType(executableNetwork, "Convert", 0);
    CheckNumberOfNodesWithTypes(executableNetwork, {"Eltwise", "Subgraph"}, 1);
}
} // namespace CPULayerTestsDefinitions
