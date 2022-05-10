// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/nonzero.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"

namespace LayerTestsDefinitions {

std::string NonZeroLayerTest::getTestCaseName(const testing::TestParamInfo<NonZeroLayerTestParamsSet>& obj) {
    ov::test::InputShape inputShape;
    InferenceEngine::Precision inputPrecision;
    std::string targetDevice;
    ConfigMap additionalConfig;
    std::tie(inputShape, inputPrecision, targetDevice, additionalConfig) = obj.param;

    std::ostringstream result;
    result << "IS=";
    result << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
    result << "TS=(";
    for (const auto& shape : inputShape.second) {
        result << CommonTestUtils::vec2str(shape) << "_";
    }
    result << ")_";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void NonZeroLayerTest::SetUp() {
    ov::test::InputShape inputShape;
    InferenceEngine::Precision inputPrecision;
    ConfigMap additionalConfig;
    std::tie(inputShape, inputPrecision, targetDevice, additionalConfig) = GetParam();

    configuration.insert(additionalConfig.cbegin(), additionalConfig.cend());
    init_input_shapes({inputShape});

    const auto& precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    const auto& paramNode = ngraph::builder::makeDynamicParams(precision, {inputDynamicShapes[0]});
    auto nonZeroOp = std::make_shared<ngraph::opset3::NonZero>(paramNode.front());

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nonZeroOp)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{paramNode}, "non_zero");
}

void NonZeroLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes)
{
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (int i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                    targetInputStaticShapes[i],
                                                                    range,
                                                                    startFrom);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}
}  // namespace LayerTestsDefinitions
