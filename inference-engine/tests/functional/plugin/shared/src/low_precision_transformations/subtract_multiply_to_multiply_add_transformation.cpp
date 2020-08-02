// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/subtract_multiply_to_multiply_add_transformation.hpp"

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
    LayerTestsUtils::LayerTransformation::LptVersion version;
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, version, testValues) = obj.param;

    std::ostringstream result;
    result <<
        targetDevice << "_" <<
        version << "_" <<
        testValues.inputShape << "_" <<
        testValues.precision << "_" <<
        testValues.fqOnData;
    return result.str();
}

void SubtractMultiplyToMultiplyAddTransformation::SetUp() {
    LayerTestsUtils::LayerTransformation::LptVersion version;
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, version, testValues) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::SubtractMultiplyToMultiplyAddFunction::getOriginal(
        testValues.inputShape,
        testValues.precision,
        testValues.fqOnData);

    if (version == LptVersion::cnnNetwork) {
        validateCNNNetwork();
    } else {
        validateNGraph();
    }
}

void SubtractMultiplyToMultiplyAddTransformation::validateNGraph() {
    LayerTestsUtils::LayerTransformation::LptVersion version;
    SubtractMultiplyToMultiplyAddTransformationTestValues testValues;
    std::tie(targetDevice, version, testValues) = this->GetParam();

    const ngraph::pass::low_precision::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams();
    auto transformed = transformNGraph(params);

    ASSERT_EQ(1ul, transformed->get_output_size());
    std::shared_ptr<ngraph::Node> output = transformed->get_output_op(0);
    std::shared_ptr<ngraph::Node> scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShift", typeName);
}

void SubtractMultiplyToMultiplyAddTransformation::validateCNNNetwork() {
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
