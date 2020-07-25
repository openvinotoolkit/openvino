// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/relu_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/relu_function.hpp"

namespace LayerTestsDefinitions {

std::string ReluTransformation::getTestCaseName(testing::TestParamInfo<ReluTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, inputShape, targetDevice, version, fqOnData) = obj.param;

    std::ostringstream result;
    result << version << "_" <<
        precision << "_" <<
        targetDevice << "_" <<
        fqOnData;

    return result.str();
}

InferenceEngine::Blob::Ptr ReluTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, inputShape, targetDevice, version, fqOnData) = this->GetParam();

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0],
        fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0],
        1ul);
}

void ReluTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, inputShape, targetDevice, version, fqOnData) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::ReluFunction::getOriginal(inputShape, precision, fqOnData);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void ReluTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    std::tie(precision, inputShape, targetDevice, version, fqOnData) = this->GetParam();

    const auto params = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);

    if (fqOnData.empty() ||
        (fqOnData.isSigned() && ((fqOnData.outputLowValues[0] / fqOnData.outputHighValues[0]) != (-128.f/127.f))) ||
        (!fqOnData.isSigned() && ((fqOnData.outputLowValues[0] != 0.f)))) {
        EXPECT_EQ("ReLU", outputLayer->type);
    } else {
        EXPECT_EQ("ScaleShift", outputLayer->type);

        EXPECT_EQ(1ul, outputLayer->insData.size());
        const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
        EXPECT_TRUE(insData != nullptr);
        const InferenceEngine::CNNLayerPtr relu = getCreatorLayer(insData).lock();
        EXPECT_TRUE(relu != nullptr);
        EXPECT_EQ("ReLU", relu->type);

        if (params.updatePrecisions) {
            const InferenceEngine::Precision precision = relu->outData[0]->getTensorDesc().getPrecision();
            EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
        }
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ReluTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
