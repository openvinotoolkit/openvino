// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/constant.hpp"

namespace LayerTestsDefinitions {
namespace {
template <size_t N>
std::vector<std::string> getElements(const std::vector<std::string>& v) {
    const auto new_size = std::min(N, v.size());
    return {begin(v), std::next(begin(v), new_size)};
}
}  // namespace

std::string ConstantLayerTest::getTestCaseName(
    const testing::TestParamInfo<constantParamsTuple>& obj) {
    std::vector<size_t> data_shape;
    InferenceEngine::Precision data_precision;
    std::vector<std::string> data_elements;
    std::string targetName;

    std::tie(data_shape, data_precision, data_elements, targetName) = obj.param;

    std::ostringstream result;
    result << "S=" << CommonTestUtils::vec2str(data_shape) << "_";
    result << "dataPRC=" << data_precision.name() << "_";
    result << "dataValue=" << CommonTestUtils::vec2str(getElements<5>(data_elements)) << "_";
    return result.str();
}

void ConstantLayerTest::SetUp() {
    std::vector<size_t> data_shape;
    InferenceEngine::Precision data_precision;
    std::vector<std::string> data_elements;

    std::tie(data_shape, data_precision, data_elements, targetDevice) = this->GetParam();

    const auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(data_precision);
    auto constant = ngraph::op::Constant::create(precision, data_shape, data_elements);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(constant)};

    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{}, "constant");
}
}  // namespace LayerTestsDefinitions
