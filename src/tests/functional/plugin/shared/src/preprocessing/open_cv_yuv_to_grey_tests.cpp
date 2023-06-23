// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <opencv2/imgproc/types_c.h>

#include <opencv2/imgproc.hpp>

// #include "openvino/op/add.hpp"
// #include "openvino/op/constant.hpp"
// #include "openvino/op/parameter.hpp"
// #include "openvino/op/result.hpp"
// #include "openvino/op/util/attr_types.hpp"
#include "preprocessing/open_cv_yuv_to_grey_tests.hpp"
#include "shared_test_classes/single_layer/convert_color_i420.hpp"
#include "shared_test_classes/single_layer/convert_color_nv12.hpp"

namespace ov {
namespace {
std::shared_ptr<ov::Model> build_test_model(const element::Type_t et, const Shape& shape) {
    const auto input = std::make_shared<op::v0::Parameter>(et, shape);
    const auto zero = op::v0::Constant::create(et, Shape{}, {0.0f});
    const auto op = std::make_shared<op::v1::Add>(input, zero);
    const auto res = std::make_shared<op::v0::Result>(op);
    return std::make_shared<ov::Model>(res, ParameterVector{input});
}
}  // namespace

namespace preprocess {
std::string PreprocessingYUV2GreyTest::getTestCaseName(const testing::TestParamInfo<TParams>& obj) {
    std::ostringstream result;
    result << "device=" << std::get<0>(obj.param);
    return result.str();
}

void PreprocessingYUV2GreyTest::SetUp() {
    const auto& test_params = GetParam();
    targetDevice = std::get<0>(test_params);

    height = width = 64;
    b_step = 5;
    outType = inType = element::u8;
}

void PreprocessingYUV2GreyTest::run() {
    compile_model();
    infer();
    validate();
}

ov::TensorVector PreprocessingYUV2GreyTest::calculate_refs() {
    return ref_out_data;
}

size_t PreprocessingYUV2GreyTest::get_full_height() {
    return height * (255 / b_step + 1);
}

void PreprocessingYUV2GreyTest::test_model_color_conversion(ColorFormat from, ColorFormat to) {
    auto ppp = PrePostProcessor(function);
    ppp.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
    ppp.input().preprocess().convert_color(ColorFormat::GRAY);
    function = ppp.build();
}

TEST_P(PreprocessingYUV2GreyTest, convert_nv12_gray) {
    // Test various possible r/g/b values within dimensions
    const auto input_y_shape = Shape{1, get_full_height(), width, 1};
    const auto input_uv_shape = Shape{1, get_full_height() / 2, width / 2, 2};
    auto ov20_input_yuv = LayerTestsDefinitions::NV12TestUtils::color_test_image(height, width, b_step);
    auto ov20_input_y =
        std::vector<uint8_t>(ov20_input_yuv.begin(), ov20_input_yuv.begin() + shape_size(input_y_shape));
    auto ov20_input_uv = std::vector<uint8_t>(ov20_input_yuv.begin() + shape_size(input_y_shape), ov20_input_yuv.end());

    // Calculate reference expected values from OpenCV
    cv::Mat pic_yv12 =
        cv::Mat(static_cast<int>(get_full_height()) * 3 / 2, static_cast<int>(width), CV_8UC1, ov20_input_yuv.data());
    // Note: cv::cvtColorTwoPlane doesn't support YUV2GRAY conversion, and
    // cv::cvtColorTwoPlane(YUV2RGB) + cv::cvtColor(RGB2GRAY) lead to huge
    // difference in U8, so compare OV TWO_PLANES vs CV SINGLE_PLANE
    cv::Mat pic_gray;
    cv::cvtColor(pic_yv12, pic_gray, CV_YUV2GRAY_NV12);
    ref_out_data.emplace_back(outType, input_y_shape, pic_gray.data);

    function = build_test_model(inType, input_y_shape);
    test_model_color_conversion(ColorFormat::NV12_TWO_PLANES, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.clear();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), input_y_shape, ov20_input_y.data()});
    inputs.emplace(params.at(1), ov::Tensor{params.at(1)->get_element_type(), input_uv_shape, ov20_input_uv.data()});

    run();
}
}  // namespace preprocess
}  // namespace ov
