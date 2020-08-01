// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/subtract_multiply_to_multiply_add.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/subtract_multiply_to_multiply_add_function.hpp"

namespace LayerTestsDefinitions {

std::string SubtractMultiplyToMultiplyAddTransformation::getTestCaseName(testing::TestParamInfo<SubtractMultiplyToMultiplyAddTransformationParams> obj) {
    std::string targetDevice;
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        targetDevice << "_" <<
        testValues.inputShape << "_" <<
        testValues.precision << "_" <<
        testValues.fqOnData;
    return result.str();
}

void SubtractMultiplyToMultiplyAddTransformation::SetUp() {
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    ConfigurePlugin(LptVersion::nGraph);

    function = ngraph::builder::subgraph::SubtractMultiplyToMultiplyAddFunction::getOriginal(
        testValues.inputShape,
        testValues.precision,
        testValues.fqOnData);

    IE_SUPPRESS_DEPRECATED_START

    const ngraph::pass::low_precision::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams();
    const InferenceEngine::CNNNetwork network = transform(toCNNNetwork(params));

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr scaleShift = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(scaleShift != nullptr);
    EXPECT_EQ("ScaleShift", scaleShift->type);
    EXPECT_EQ(InferenceEngine::Precision::FP32, scaleShift->outData[0]->getPrecision());

    const InferenceEngine::CNNLayerPtr reshape = getCreatorLayer(scaleShift->insData[0].lock()).lock();
    EXPECT_EQ("Reshape", reshape->type);
    EXPECT_EQ(params.updatePrecisions ? InferenceEngine::Precision::U8 : InferenceEngine::Precision::FP32, reshape->outData[0]->getPrecision());

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(SubtractMultiplyToMultiplyAddTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
