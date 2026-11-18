// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/convert_color_to_nv12.hpp"

#include "openvino/op/bgr_to_nv12.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/rgb_to_nv12.hpp"

namespace ov {
namespace test {
std::string ConvertColorToNV12LayerTest::getTestCaseName(
    const testing::TestParamInfo<ConvertColorToNV12ParamsTuple>& obj) {
    const auto& [shapes, type, conversion, single_plane, device_name] = obj.param;
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

void ConvertColorToNV12LayerTest::SetUp() {
    const auto& [shapes, net_type, conversionToRGB, single_plane, _targetDevice] = GetParam();
    targetDevice = _targetDevice;
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(net_type, inputDynamicShapes.front());

    std::shared_ptr<ov::Node> convert_color;
    if (conversionToRGB) {
        convert_color = std::make_shared<ov::op::v17::RGBtoNV12>(param, single_plane);
    } else {
        convert_color = std::make_shared<ov::op::v17::BGRtoNV12>(param, single_plane);
    }
    function = std::make_shared<ov::Model>(convert_color->outputs(), ov::ParameterVector{param}, "ConvertColorToNV12");
}
}  // namespace test
}  // namespace ov
