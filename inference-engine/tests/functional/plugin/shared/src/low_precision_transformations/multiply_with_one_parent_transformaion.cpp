// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_with_one_parent_transformation.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ie_core.hpp>
#include "common_test_utils/common_utils.hpp"
#include "ngraph_functions/low_precision_transformations/multiply_with_one_parent_function.hpp"

namespace LayerTestsDefinitions {

std::string MultiplyWithOneParentTransformation::getTestCaseName(testing::TestParamInfo<MultiplyWithOneParentTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    MultiplyWithOneParentTransformationValues values;

    std::tie(netPrecision, inputShape, targetDevice, values) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << CommonTestUtils::vec2str(inputShape);
    return result.str();
}

void MultiplyWithOneParentTransformation::SetUp() {
    threshold = 0.01f;

    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::details::LayerTransformation::Params params;
    MultiplyWithOneParentTransformationValues values;
    std::tie(netPrecision, inputShape, targetDevice, values) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    function = ngraph::builder::subgraph::MultiplyWithOneParentFunction::getOriginal(precision, inputShape, values.fakeQuantize);

    validate();
}

void MultiplyWithOneParentTransformation::validate() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    MultiplyWithOneParentTransformationValues values;
    std::tie(netPrecision, inputShape, targetDevice, values) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("Eltwise", outputLayer->type);

    // check #1: successful transformation execution
    EXPECT_EQ(2ul, outputLayer->insData.size());
    const auto parents = InferenceEngine::details::CNNNetworkHelper::getParents(*outputLayer);
    EXPECT_EQ(2ul, parents.size());
    EXPECT_EQ("ScaleShift", parents[0]->type);

    // check #2: successful graph handling
    EXPECT_EQ("FakeQuantize", parents[1]->type);
    EXPECT_EQ(1ul, InferenceEngine::details::CNNNetworkHelper::getParents(*parents[0]).size());
    EXPECT_EQ("FakeQuantize", InferenceEngine::details::CNNNetworkHelper::getParents(*parents[0])[0]->type);

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(MultiplyWithOneParentTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
