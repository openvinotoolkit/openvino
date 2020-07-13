// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_max_pool_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
//#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/max_pool_function.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndMaxPoolTransformation::getTestCaseName(testing::TestParamInfo<FakeQuantizeAndMaxPoolTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShapes, targetDevice, params, version, fakeQuantize) = obj.param;

    return getTestCaseNameByParams(precision, inputShapes, targetDevice, params, version);
}

void FakeQuantizeAndMaxPoolTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShape, targetDevice, params, version, fakeQuantize) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::MaxPoolFunction::getOriginal(
        precision,
        inputShape,
        fakeQuantize);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void FakeQuantizeAndMaxPoolTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShape, targetDevice, params, version, fakeQuantize) = this->GetParam();

    const auto cnnnetworkParams = toCNNNetwork(params);
    const InferenceEngine::CNNNetwork network = transform(cnnnetworkParams);

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
    const InferenceEngine::CNNLayerPtr pooling = getCreatorLayer(insData).lock();
    EXPECT_TRUE(pooling != nullptr);
    EXPECT_EQ("Pooling", pooling->type);

    if (params.updatePrecisions) {
        const InferenceEngine::Precision precision = pooling->outData[0]->getTensorDesc().getPrecision();
        EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(FakeQuantizeAndMaxPoolTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
