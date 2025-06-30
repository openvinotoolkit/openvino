// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/convert_color_nv12.hpp"

#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
std::string ConvertColorNV12LayerTest::getTestCaseName(const testing::TestParamInfo<ConvertColorNV12ParamsTuple> &obj) {
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
    result << "modelType=" << type.c_type_string() << "_";
    result << "convRGB=" << conversion << "_";
    result << "single_plane=" << single_plane << "_";
    result << "targetDevice=" << device_name;
    return result.str();
}

void ConvertColorNV12LayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type net_type;
    bool conversionToRGB, single_plane;
    abs_threshold = 1.1f; // NV12 conversion can use various algorithms, thus some absolute deviation is allowed
    rel_threshold = 1.1f; // Ignore relative comparison for NV12 convert (allow 100% relative deviation)
    std::tie(shapes, net_type, conversionToRGB, single_plane, targetDevice) = GetParam();
    init_input_shapes(shapes);

    if (single_plane) {
        auto param = std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes.front());

        std::shared_ptr<ov::Node> convert_color;
        if (conversionToRGB) {
            convert_color = std::make_shared<ov::op::v8::NV12toRGB>(param);
        } else {
            convert_color = std::make_shared<ov::op::v8::NV12toBGR>(param);
        }
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                      ov::ParameterVector{param}, "ConvertColorNV12");
    } else {
        auto param_y  = std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes[0]);
        auto param_uv = std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes[1]);

        std::shared_ptr<ov::Node> convert_color;
        if (conversionToRGB) {
            convert_color = std::make_shared<ov::op::v8::NV12toRGB>(param_y, param_uv);
        } else {
            convert_color = std::make_shared<ov::op::v8::NV12toBGR>(param_y, param_uv);
        }
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                      ov::ParameterVector{param_y, param_uv}, "ConvertColorNV12");
    }
}
} // namespace test
} // namespace ov
