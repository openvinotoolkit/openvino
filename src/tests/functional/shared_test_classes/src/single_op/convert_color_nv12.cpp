// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/convert_color_nv12.hpp"

#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace {
template <typename T>
inline void validate_colors(const T* expected, const T* actual, size_t size, float dev_threshold, float abs_threshold = 0.01f) {
    size_t mismatches = 0;
    for (size_t i = 0; i < size; i++) {
        if (std::abs(static_cast<float>(expected[i]) - static_cast<float>(actual[i])) > abs_threshold) {
            mismatches++;
        }
    }
    ASSERT_LT(static_cast<float>(mismatches) / size, dev_threshold) << mismatches <<
        " out of " << size << " color mismatches found which exceeds allowed threshold " << dev_threshold;
}

inline std::vector<uint8_t> color_test_image(size_t height, size_t width, int b_step) {
    // Test all possible r/g/b values within dimensions
    int b_dim = 255 / b_step + 1;
    auto input_yuv = std::vector<uint8_t>(height * b_dim * width * 3 / 2);
    for (int b = 0; b <= 255; b += b_step) {
        for (size_t y = 0; y < height / 2; y++) {
            for (size_t x = 0; x < width / 2; x++) {
                int r = static_cast<int>(y) * 512 / static_cast<int>(height);
                int g = static_cast<int>(x) * 512 / static_cast<int>(width);
                // Can't use random y/u/v for testing as this can lead to invalid R/G/B values
                int y_val = ((66 * r + 129 * g + 25 * b + 128) / 256) + 16;
                int u_val = ((-38 * r - 74 * g + 112 * b + 128) / 256) + 128;
                int v_val = ((112 * r - 94 * g + 18 * b + 128) / 256) + 128;

                size_t b_offset = height * width * b / b_step * 3 / 2;
                size_t uv_index = b_offset + height * width + y * width + x * 2;
                input_yuv[uv_index] = u_val;
                input_yuv[uv_index + 1] = v_val;
                size_t y_index = b_offset + y * 2 * width + x * 2;
                input_yuv[y_index] = y_val;
                input_yuv[y_index + 1] = y_val;
                input_yuv[y_index + width] = y_val;
                input_yuv[y_index + width + 1] = y_val;
            }
        }
    }
    return input_yuv;
}

void generate_tensor(ov::Tensor& tensor) {
    size_t full_height = tensor.get_shape()[1];
    size_t full_width = tensor.get_shape()[2];
    int b_dim = static_cast<int>(full_height * 2 / (3 * full_width));
    ASSERT_GT(b_dim, 1) << "Image height is invalid for NV12 Accuracy test";
    ASSERT_EQ(255 % (b_dim - 1), 0) << "Image height is invalid for NV12 Accuracy test";
    int b_step = 255 / (b_dim - 1);
    auto input_image = color_test_image(full_width, full_width, b_step);
    auto data_ptr = static_cast<uint8_t*>(tensor.data());
    for (size_t j = 0; j < input_image.size(); ++j) {
        data_ptr[j] = input_image[j];
    }
}
} // namespace

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
    result << "netPRC=" << type.c_type_string() << "_";
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

        //inputShape[1] = inputShape[1] * 3 / 2;
        std::shared_ptr<ov::Node> convert_color;
        if (conversionToRGB) {
            convert_color = std::make_shared<ov::op::v8::NV12toRGB>(param);
        } else {
            convert_color = std::make_shared<ov::op::v8::NV12toBGR>(param);
        }
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                      ov::ParameterVector{param}, "ConvertColorNV12");
    } else {
        //auto uvShape = ov::Shape{inputShape[0], inputShape[1] / 2, inputShape[2] / 2, 2};
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

// -------- Accuracy test (R/G/B combinations) --------

void ConvertColorNV12AccuracyTest::generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) {
    inputs.clear();
    auto params = function->get_parameters();
    OPENVINO_ASSERT(target_input_static_shapes.size() >= params.size());
    for (int i = 0; i < params.size(); i++) {
        ov::Tensor tensor(params[i]->get_element_type(), target_input_static_shapes[i]);
        generate_tensor(tensor);
        inputs.insert({params[i], tensor});
    }
}

void ConvertColorNV12AccuracyTest::compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) {
    ConvertColorNV12LayerTest::compare(expected, actual);

    ASSERT_FALSE(expected.empty());
    ASSERT_FALSE(actual.empty());

    // Allow less than 2% of deviations with 1 color step. 2% is experimental value
    // For different calculation methods - 1.4% deviation is observed
    for (int i = 0; i < expected.size(); i++) {
        validate_colors(static_cast<float*>(expected[i].data()), static_cast<float*>(actual[i].data()), expected.size(), 0.02);
    }
}
} // namespace test
} // namespace ov
