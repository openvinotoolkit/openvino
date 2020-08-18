// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/convert_pad_to_group_conv.hpp"

namespace LayerTestsDefinitions {

std::string ConvertPadToConvTests::getTestCaseName(const testing::TestParamInfo<PadParams> &obj) {
    ngraph::Shape input_shape;
    std::string targetName;
    std::vector<int64_t> pad_begin, pad_end;
    ngraph::op::PadMode mode;
    float value;
    std::tie(input_shape, pad_begin, pad_end, value, mode, targetName) = obj.param;
    std::ostringstream results;

    results << "Input" << CommonTestUtils::vec2str(input_shape);
    results << "PadBegin" << CommonTestUtils::vec2str(pad_begin);
    results << "PadEnd" << CommonTestUtils::vec2str(pad_end);
    results << "Value" << value;
    results << "Mode" << mode;
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void ConvertPadToConvTests::SetUp() {
    ngraph::Shape input_shape;
    std::vector<int64_t> pad_begin, pad_end;
    ngraph::op::PadMode mode;
    float value;
    std::tie(input_shape, pad_begin, pad_end, value, mode, targetDevice) = this->GetParam();

    {
        auto param = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, input_shape);
        auto pad = std::make_shared<ngraph::opset4::Pad>(param,
                                                         ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{pad_begin.size()}, pad_begin),
                                                         ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{pad_end.size()}, pad_end),
                                                         ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {value}), mode);
        auto relu = std::make_shared<ngraph::opset4::Relu>(pad);
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{relu}, ngraph::ParameterVector{param}, "pad");
    }
}

TEST_P(ConvertPadToConvTests, CompareWithRefs) {
    Run();
}
} // namespace LayerTestsDefinitions
