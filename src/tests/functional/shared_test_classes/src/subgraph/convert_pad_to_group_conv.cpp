// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/convert_pad_to_group_conv.hpp"

namespace ov {
namespace test {

std::string ConvertPadToConvTests::getTestCaseName(const testing::TestParamInfo<PadParams>& obj) {
    ov::Shape input_shape;
    std::string targetName;
    std::vector<int64_t> pad_begin, pad_end;
    ov::op::PadMode mode;
    float value;
    std::tie(input_shape, pad_begin, pad_end, value, mode, targetName) = obj.param;
    std::ostringstream results;

    results << "Input" << ov::test::utils::vec2str(input_shape);
    results << "PadBegin" << ov::test::utils::vec2str(pad_begin);
    results << "PadEnd" << ov::test::utils::vec2str(pad_end);
    results << "Value" << value;
    results << "Mode" << mode;
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void ConvertPadToConvTests::SetUp() {
    ov::Shape input_shape;
    std::vector<int64_t> pad_begin, pad_end;
    ov::op::PadMode mode;
    float value;
    std::tie(input_shape, pad_begin, pad_end, value, mode, targetDevice) = this->GetParam();

    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        auto pad = std::make_shared<ov::op::v1::Pad>(
            param,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{pad_begin.size()}, pad_begin),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{pad_end.size()}, pad_end),
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {value}),
            mode);
        auto relu = std::make_shared<ov::op::v0::Relu>(pad);
        function = std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{param}, "pad");
    }
}

}  // namespace test
}  // namespace ov
