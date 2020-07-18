// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/normalize_transformation.hpp"

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
#include "ngraph_functions/low_precision_transformations/normalize_l2_function.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeL2Transformation::getTestCaseName(testing::TestParamInfo<NormalizeL2TransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::vector<uint64_t> axes;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, shapes, targetDevice, version, axes, fuseMultiply, shift) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" <<
        shapes.first << "_" <<
        shapes.second << "_" <<
        targetDevice << "_" <<
        toString(params) << "_" <<
        version <<
        "_axes" << axes.size() <<
        (fuseMultiply ? "_multiply" : "") <<
        (shift ? "_shift" : "");
    return result.str();
}

void NormalizeL2Transformation::SetUp() {
    threshold = 3.e-3;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::vector<uint64_t> axes;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, shapes, targetDevice, version, axes, fuseMultiply, shift) = this->GetParam();

    ConfigurePlugin(version);

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    function = ngraph::builder::subgraph::NormalizeL2Function::getOriginal(
        ngPrc,
        shapes,
        FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(params.precisionsOnActivations[0]),
        axes,
        fuseMultiply,
        shift);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void NormalizeL2Transformation::validate() {
    InferenceEngine::Precision netPrecision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    InferenceEngine::details::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::vector<uint64_t> axes;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, shapes, targetDevice, version, axes, fuseMultiply, shift) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ(shift ? "Normalize" : "ScaleShift", outputLayer->type);

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(NormalizeL2Transformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
