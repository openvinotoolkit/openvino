// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mvn_transformation.hpp"

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
#include "ngraph_functions/low_precision_transformations/mvn_function.hpp"

#include <ngraph/pass/visualize_tree.hpp>

namespace LayerTestsDefinitions {

std::string MVNTransformation::getTestCaseName(testing::TestParamInfo<MVNTransformationParams> obj) {
    std::string targetDevice;
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::AxisSet reductionAxes;
    bool normalizeVariance;
    std::tie(precision, shape, targetDevice, version, reductionAxes, normalizeVariance) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, shape, targetDevice, params, version) <<
        "_" << reductionAxes << "_" << normalizeVariance;
    return result.str();
}

void MVNTransformation::SetUp() {
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::AxisSet reductionAxes;
    bool normalizeVariance;
    std::tie(precision, shape, targetDevice, version, reductionAxes, normalizeVariance) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::MVNFunction::getOriginal(
        precision,
        shape,
        reductionAxes,
        normalizeVariance);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void MVNTransformation::validate() {
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::AxisSet reductionAxes;
    bool normalizeVariance;
    std::tie(precision, shape, targetDevice, version, reductionAxes, normalizeVariance) = this->GetParam();

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
    const InferenceEngine::CNNLayerPtr mvn = getCreatorLayer(insData).lock();
    EXPECT_TRUE(mvn != nullptr);
    EXPECT_EQ("MVN", mvn->type);

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(MVNTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
