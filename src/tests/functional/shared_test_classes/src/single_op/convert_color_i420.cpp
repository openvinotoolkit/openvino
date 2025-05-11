// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/convert_color_i420.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/i420_to_bgr.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
std::string ConvertColorI420LayerTest::getTestCaseName(const testing::TestParamInfo<ConvertColorI420ParamsTuple> &obj) {
    std::vector<InputShape> shapes;
    ov::element::Type type;
    bool conversion, single_plane;
    std::string device_name;
    std::tie(shapes, type, conversion, single_plane, device_name) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "netPRC=" << type.c_type_string() << "_";
    result << "convRGB=" << conversion << "_";
    result << "singlePlane=" << single_plane << "_";
    result << "targetDevice=" << device_name;
    return result.str();
}

void ConvertColorI420LayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type net_type;
    bool conversion_to_rgb;
    bool single_plane;
    abs_threshold = 1.1f; // I420 conversion can use various algorithms, thus some absolute deviation is allowed
    rel_threshold = 1.1f; // Ignore relative comparison for I420 convert (allow 100% relative deviation)
    std::tie(shapes, net_type, conversion_to_rgb, single_plane, targetDevice) = GetParam();
    init_input_shapes(shapes);

    if (single_plane) {
        auto param = std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes.front());
        std::shared_ptr<ov::Node> convert_color;
        if (conversion_to_rgb) {
            convert_color = std::make_shared<ov::op::v8::I420toRGB>(param);
        } else {
            convert_color = std::make_shared<ov::op::v8::I420toBGR>(param);
        }
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                      ov::ParameterVector{param}, "ConvertColorI420");
    } else {
        auto param_y = std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes[0]);
        auto param_u = std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes[1]);
        auto param_v = std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes[2]);
        std::shared_ptr<ov::Node> convert_color;
        if (conversion_to_rgb) {
            convert_color = std::make_shared<ov::op::v8::I420toRGB>(param_y, param_u, param_v);
        } else {
            convert_color = std::make_shared<ov::op::v8::I420toBGR>(param_y, param_u, param_v);
        }
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                      ov::ParameterVector{param_y, param_u, param_v},
                                                      "ConvertColorI420");
    }
}
} // namespace test
} // namespace ov
