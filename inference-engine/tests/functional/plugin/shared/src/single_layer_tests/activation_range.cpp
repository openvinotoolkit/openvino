// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/activation_range.hpp"

namespace LayerTestsDefinitions {

std::string ActivationLayerRangeTest::getTestCaseName(const testing::TestParamInfo<activationRangeParams> &obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::string targetDevice;
    std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationDecl;
    std::pair<int32_t, uint32_t> inValRange;
    std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, inValRange, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << activationNames[activationDecl.first] << separator;
    result << "IS=" << CommonTestUtils::vec2str(shapes.first) << separator;
    result << "AS=" << CommonTestUtils::vec2str(shapes.second) << separator;
    result << "ConstantsValue=" << CommonTestUtils::vec2str(activationDecl.second) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "inPRC=" << inPrc.name() << separator;
    result << "outPRC=" << outPrc.name() << separator;
    result << "inL=" << inLayout << separator;
    result << "outL=" << outLayout << separator;
    result << "min=";
    if (inValRange.first < 0) {
        result << "minus";
    }
    result << std::abs(inValRange.first) << "range=" << inValRange.second;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ActivationLayerRangeTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, inputValuesRange, targetDevice) = GetParam();

    activationType = activationDecl.first;
    auto constantsValue = activationDecl.second;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {shapes.first});
    auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType, shapes.second, constantsValue);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params);
}

InferenceEngine::Blob::Ptr ActivationLayerRangeTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    bool inPrcSigned = function->get_parameters()[0]->get_element_type().is_signed();
    int32_t data_start_from = inputValuesRange.first;
    uint32_t data_range = inputValuesRange.second;
    if (!inPrcSigned) {
        data_start_from = 0;
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_range,
                                            data_start_from,
                                            32768);
}

TEST_P(ActivationLayerRangeTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
