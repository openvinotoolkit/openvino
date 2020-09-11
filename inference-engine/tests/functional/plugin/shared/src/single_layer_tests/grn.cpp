// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/grn.hpp"

namespace LayerTestsDefinitions {
std::string GrnLayerTest::getTestCaseName(const testing::TestParamInfo<grnParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    float bias;
    std::tie(netPrecision,
        inputShapes,
        bias,
        targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';

    result << "IS="     << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "bias="   << bias << separator;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void GrnLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShapes, bias, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { inputShapes });
    auto paramsOut = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto grn = std::make_shared<ngraph::opset1::GRN>(paramsOut[0], bias);
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(grn) };
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Grn");
}

TEST_P(GrnLayerTest, CompareWithRefs) {
    Run();
};
}  // namespace LayerTestsDefinitions
