
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "subgraph_tests/codegen_convert.hpp"

namespace LayerTestsDefinitions {

std::string CodegenConvert::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::convertParams> obj) {
    ov::element::Type inType;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(inType, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << inType.get_type_name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void CodegenConvert::SetUp() {
    InferenceEngine::SizeVector inputShape;
    ov::element::Type inType;
    std::tie(inType, inputShape, targetDevice) = this->GetParam();

    auto input = std::make_shared<ngraph::opset1::Parameter>(inType, ngraph::Shape{inputShape});

    std::vector<int64_t> bpads(inputShape.size(), 0);
    std::vector<int64_t> epads(inputShape.size(), 0);
    epads[0] = 10;
    auto pad = ngraph::builder::makePad(input, bpads, epads, 0.0, ngraph::helpers::PadMode::CONSTANT);
    auto relu = std::make_shared<ngraph::opset1::Relu>(pad);
    auto constant = ngraph::opset1::Constant::create(inType, ngraph::Shape{}, {10});
    auto add = std::make_shared<ngraph::opset1::Add>(relu, constant);
    auto slice = ngraph::builder::makeSlice(add, {0}, {5}, {1}, {0}, inType);
    auto result = std::make_shared<ngraph::opset1::Result>(slice);

    function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{result},
        ngraph::ParameterVector{input},
        "CodegenConvert");
}

TEST_P(CodegenConvert, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
