// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_to_group_convolution_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/low_precision_transformations/multiply_to_group_convolution_function.hpp"

#include <ngraph/pass/visualize_tree.hpp>

namespace LayerTestsDefinitions {

std::string MultiplyToGroupConvolutionTransformation::getTestCaseName(testing::TestParamInfo<MultiplyToGroupConvolutionTransformationParams> obj) {
    std::string targetDevice;
    ngraph::element::Type precision;
    ngraph::Shape shape;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, shape, targetDevice, version, fqOnData) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, shape, targetDevice, params, version) <<
        "_" << fqOnData;
    return result.str();
}

void MultiplyToGroupConvolutionTransformation::SetUp() {
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, shape, targetDevice, version, fqOnData) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::MultiplyToGroupConvolutionFunction::getOriginal(
        precision,
        shape,
        fqOnData);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void MultiplyToGroupConvolutionTransformation::validate() {
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, shape, targetDevice, version, fqOnData) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(toCNNNetwork(params));

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    EXPECT_EQ(1ul, outputLayer->insData.size());
    const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr);
    const InferenceEngine::CNNLayerPtr conv = getCreatorLayer(insData).lock();
    EXPECT_TRUE(conv != nullptr);
    EXPECT_EQ("Convolution", conv->type);

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(MultiplyToGroupConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
