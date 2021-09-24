// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convert_color_nv12.hpp"
namespace LayerTestsDefinitions {

std::string ConvertColorNV12LayerTest::getTestCaseName(const testing::TestParamInfo<ConvertColorNV12ParamsTuple> &obj) {
    ngraph::Shape inputShape;
    ngraph::element::Type type;
    bool conversion, singlePlane;
    std::string targetName;
    std::tie(inputShape, type, conversion, singlePlane, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC=" << type.c_type_string() << "_";
    result << "convRGB=" << conversion << "_";
    result << "singlePlane=" << singlePlane << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void ConvertColorNV12LayerTest::SetUp() {
    ngraph::Shape inputShape;
    ngraph::element::Type ngPrc;
    bool conversionToRGB, singlePlane;
    threshold = 2.0f; // NV12 color conversion can use various of algorithms, thus some deviation is allowed
    std::tie(inputShape, ngPrc, conversionToRGB, singlePlane, targetDevice) = GetParam();
    if (singlePlane) {
        inputShape[1] = inputShape[1] * 3 / 2;
        auto param = std::make_shared<ngraph::op::Parameter>(ngPrc, inputShape);
        std::shared_ptr<ov::Node> convert_color;
        if (conversionToRGB) {
            convert_color = std::make_shared<ov::op::v8::NV12toRGB>(param);
        } else {
            convert_color = std::make_shared<ov::op::v8::NV12toBGR>(param);
        }
        function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(convert_color),
                                                      ngraph::ParameterVector{param}, "ConvertColorNV12");
    } else {
        auto uvShape = ngraph::Shape{inputShape[0], inputShape[1] / 2, inputShape[2] / 2, 2};
        auto param_y = std::make_shared<ngraph::op::Parameter>(ngPrc, inputShape);
        auto param_uv = std::make_shared<ngraph::op::Parameter>(ngPrc, uvShape);
        std::shared_ptr<ov::Node> convert_color;
        if (conversionToRGB) {
            convert_color = std::make_shared<ov::op::v8::NV12toRGB>(param_y, param_uv);
        } else {
            convert_color = std::make_shared<ov::op::v8::NV12toBGR>(param_y, param_uv);
        }
        function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(convert_color),
                                                      ngraph::ParameterVector{param_y, param_uv}, "ConvertColorNV12");
    }
}

} // namespace LayerTestsDefinitions