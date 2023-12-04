// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/normalize_l2.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeL2LayerTest::getTestCaseName(const testing::TestParamInfo<NormalizeL2LayerTestParams>& obj) {
    std::vector<int64_t> axes;
    float eps;
    ngraph::op::EpsMode epsMode;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(axes, eps, epsMode, inputShape, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "eps=" << eps << "_";
    result << "epsMode=" << epsMode << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr NormalizeL2LayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::Blob::Ptr blobPtr;
    const std::string& name = info.name();
    if (name == "data") {
        blobPtr = FuncTestUtils::createAndFillBlobFloat(info.getTensorDesc(), 10, -5, 7, 222);
    } else {
        blobPtr = LayerTestsUtils::LayerTestsCommon::GenerateInput(info);
    }
    return blobPtr;
}

void NormalizeL2LayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> axes;
    float eps;
    ngraph::op::EpsMode epsMode;
    InferenceEngine::Precision netPrecision;
    std::tie(axes, eps, epsMode, inputShape, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto data_input = params[0];
    data_input->set_friendly_name("data");

    auto normAxes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
    auto norm = std::make_shared<ov::op::v0::NormalizeL2>(data_input, normAxes, eps, epsMode);

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(norm)};
    function = std::make_shared<ngraph::Function>(results, params, "NormalizeL2");
}

}  // namespace LayerTestsDefinitions
