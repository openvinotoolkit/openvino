// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_add_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/multiply_add_function.hpp"

namespace LayerTestsDefinitions {

std::string MultiplyAddTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShapes, targetDevice, params, version) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version);
}

void MultiplyAddTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    ConfigurePlugin(version);

    const auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    function = ngraph::builder::subgraph::MultiplyAddFunction::getOriginal(ngPrecision, inputShape);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void MultiplyAddTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    if (params.updatePrecisions) {
        const InferenceEngine::Precision precision = outputLayer->outData[0]->getTensorDesc().getPrecision();
        EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(MultiplyAddTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
