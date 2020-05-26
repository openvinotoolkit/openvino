// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ie_plugin_config.hpp>
#include <ie_core.hpp>
#include <functional>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "single_layer_tests/reshape.hpp"

namespace LayerTestsDefinitions {
    std::string ReshapeLayerTest::getTestCaseName(testing::TestParamInfo<reshapeParams> obj) {
    InferenceEngine::Precision inputPrecision;
    ShapeConfig shapeConfig;
    bool specialZero, synBatch;
    std::string targetDevice;

    std::tie(inputPrecision, shapeConfig, specialZero, synBatch, targetDevice) = obj.param;

    std::ostringstream result;
    result << obj.index << "_";
    result << "is=" << CommonTestUtils::vec2str(shapeConfig.first) << "_";
    result << "os=" << CommonTestUtils::vec2str(shapeConfig.second) << "_";
    result << "ip=" << inputPrecision << "_";
    result << "specZero=" << specialZero << "_";
    result << "dynBatch=" << synBatch << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void ReshapeLayerTest::SetUp() {
    InferenceEngine::Precision inputPrecision;
    ShapeConfig shapeConfig;
    bool specialZero, dynBatch;

    std::tie(inputPrecision, shapeConfig, specialZero, dynBatch, targetDevice) = this->GetParam();
    auto inShape = shapeConfig.first;
    auto outShape = shapeConfig.second;

    /* TODO: WA. The CNNNetwork loses original precision info during some internal fallback logic.
                 So we have to restore it manually.  */
    outPrc = inputPrecision;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);

    using namespace ngraph::opset1;
    using ngraph::element::Type_t;
    using ngraph::Shape;
    using std::make_shared;

    auto input1 = make_shared<Parameter>(ngPrc, Shape(inShape));
    auto input2 = make_shared<Constant>(Type_t::i64, Shape{outShape.size()}, outShape);
    auto reshape = make_shared<Reshape>(input1, input2, specialZero);
    auto result = make_shared<Result>(reshape);

    function = make_shared<ngraph::Function>(
            ngraph::ResultVector {result},
            ngraph::ParameterVector {input1},
            "Reshape");

    configuration[CONFIG_KEY(DYN_BATCH_ENABLED)] = dynBatch ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);
}

TEST_P(ReshapeLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions