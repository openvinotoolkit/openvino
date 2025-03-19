// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/common_utils.hpp"
#include "preprocessing/yuv_to_grey_tests.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"

namespace ov {
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

    inputs.clear();
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

std::shared_ptr<ov::Model> PreprocessingYUV2GreyTest::build_test_model(const element::Type_t et, const Shape& shape) {
    const auto input = std::make_shared<op::v0::Parameter>(et, shape);
    const auto zero = op::v0::Constant::create(et, Shape{}, {0.0f});
    const auto op = std::make_shared<op::v1::Add>(input, zero);
    const auto res = std::make_shared<op::v0::Result>(op);
    return std::make_shared<ov::Model>(res, ParameterVector{input});
}

void PreprocessingYUV2GreyTest::set_test_model_color_conversion(ColorFormat from, ColorFormat to) {
    auto ppp = PrePostProcessor(function);
    ppp.input().tensor().set_color_format(from);
    ppp.input().preprocess().convert_color(to);
    function = ppp.build();
}

TEST_P(PreprocessingYUV2GreyTest, convert_single_plane_i420_hardcoded_ref) {
    // clang-format off
    auto input = std::vector<uint8_t> {0x51, 0xeb, 0x51, 0xeb,
                                       0x51, 0xeb, 0x51, 0xeb,
                                       0x6d, 0x6d, 0xb8, 0xb8};

    auto exp_out = std::vector<uint8_t> {0x51, 0xeb, 0x51, 0xeb,
                                         0x51, 0xeb, 0x51, 0xeb};
    // clang-format on

    const auto yuv_input_shape = ov::Shape{1, 6, 2, 1};
    const auto output_shape = ov::Shape{1, 4, 2, 1};
    const auto input_model_shape = output_shape;

    ref_out_data.emplace_back(outType, output_shape, exp_out.data());

    // Build model and set inputs
    function = build_test_model(inType, input_model_shape);
    set_test_model_color_conversion(ColorFormat::I420_SINGLE_PLANE, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), yuv_input_shape, input.data()});

    run();
}

TEST_P(PreprocessingYUV2GreyTest, convert_three_plane_i420_hardcoded_ref) {
    // clang-format off
    auto input_y = std::vector<uint8_t> {0x51, 0xeb, 0x51, 0xeb,
                                         0x52, 0xeb, 0x51, 0xeb};
    auto input_u = std::vector<uint8_t> {0x10, 0x12};
    auto input_v = std::vector<uint8_t> {0x21, 0x22};

    auto exp_out = std::vector<uint8_t> {0x51, 0xeb, 0x51, 0xeb,
                                         0x52, 0xeb, 0x51, 0xeb};
    // clang-format on
    const auto input_y_shape = ov::Shape{1, 4, 2, 1};
    const auto input_u_shape = ov::Shape{1, 2, 1, 1};
    const auto input_v_shape = ov::Shape{1, 2, 1, 1};
    const auto output_shape = ov::Shape{1, 4, 2, 1};
    const auto input_model_shape = output_shape;

    ref_out_data.emplace_back(outType, output_shape, exp_out.data());

    // Build model and set inputs
    function = build_test_model(inType, input_model_shape);
    set_test_model_color_conversion(ColorFormat::I420_THREE_PLANES, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), input_y_shape, input_y.data()});
    inputs.emplace(params.at(1), ov::Tensor{params.at(1)->get_element_type(), input_u_shape, input_u.data()});
    inputs.emplace(params.at(2), ov::Tensor{params.at(2)->get_element_type(), input_v_shape, input_v.data()});

    run();
}

TEST_P(PreprocessingYUV2GreyTest, convert_single_nv12_plane_hardcoded_ref) {
    // clang-format off
    auto input = std::vector<uint8_t> {0x51, 0xeb, 0x51, 0xeb,
                                       0x51, 0xeb, 0x51, 0xeb,
                                       0x6d, 0xb8, 0x6d,  0xb8};

    auto exp_out = std::vector<uint8_t> {0x51, 0xeb, 0x51, 0xeb,
                                         0x51, 0xeb, 0x51, 0xeb};
    // clang-format on

    const auto yuv_input_shape = ov::Shape{1, 6, 2, 1};
    const auto output_shape = ov::Shape{1, 4, 2, 1};
    const auto input_model_shape = output_shape;

    ref_out_data.emplace_back(outType, output_shape, exp_out.data());

    // Build model and set inputs
    function = build_test_model(inType, input_model_shape);
    set_test_model_color_conversion(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), yuv_input_shape, input.data()});

    run();
}

TEST_P(PreprocessingYUV2GreyTest, convert_two_plane_nv12_hardcoded_ref) {
    // clang-format off
    auto input_y = std::vector<uint8_t> {0x51, 0xeb, 0x51, 0xeb,
                                         0x52, 0xeb, 0x51, 0xeb};
    auto input_uv = std::vector<uint8_t> {0x10, 0x12, 0x21, 0x22};

    auto exp_out = std::vector<uint8_t> {0x51, 0xeb, 0x51, 0xeb,
                                         0x52, 0xeb, 0x51, 0xeb};
    // clang-format on
    const auto input_y_shape = ov::Shape{1, 4, 2, 1};
    const auto input_uv_shape = ov::Shape{1, 2, 1, 2};
    const auto output_shape = ov::Shape{1, 4, 2, 1};
    const auto input_model_shape = output_shape;

    ref_out_data.emplace_back(outType, output_shape, exp_out.data());

    // Build model and set inputs
    function = build_test_model(inType, input_model_shape);
    set_test_model_color_conversion(ColorFormat::NV12_TWO_PLANES, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), input_y_shape, input_y.data()});
    inputs.emplace(params.at(1), ov::Tensor{params.at(1)->get_element_type(), input_uv_shape, input_uv.data()});

    run();
}

TEST_P(PreprocessingYUV2GreyTest, convert_single_plane_i420_use_opencv) {
    // Test various possible r/g/b values within dimensions
    const auto input_yuv_shape = Shape{1, get_full_height() * 3 / 2, width, 1};
    const auto input_y_shape = Shape{1, get_full_height(), width, 1};
    auto ov20_input_yuv = ov::test::utils::color_test_image(height, width, b_step, ColorFormat::I420_SINGLE_PLANE);
    auto ov20_input_y =
        std::vector<uint8_t>(ov20_input_yuv.begin(), ov20_input_yuv.begin() + shape_size(input_y_shape));

    ref_out_data.emplace_back(outType, input_y_shape, ov20_input_y.data());

    // Build model and set inputs
    function = build_test_model(inType, input_y_shape);
    set_test_model_color_conversion(ColorFormat::I420_SINGLE_PLANE, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), input_yuv_shape, ov20_input_yuv.data()});

    run();
}

TEST_P(PreprocessingYUV2GreyTest, convert_three_plane_i420_use_opencv) {
    // Test various possible r/g/b values within dimensions
    const auto input_y_shape = Shape{1, get_full_height(), width, 1};
    const auto input_u_shape = Shape{1, get_full_height() / 2, width / 2, 1};
    const auto input_v_shape = Shape{1, get_full_height() / 2, width / 2, 1};
    auto ov20_input_yuv = ov::test::utils::color_test_image(height, width, b_step, ColorFormat::I420_THREE_PLANES);

    auto input_yuv_iter = ov20_input_yuv.begin();
    auto ov20_input_y = std::vector<uint8_t>(input_yuv_iter, input_yuv_iter + shape_size(input_y_shape));

    input_yuv_iter += shape_size(input_y_shape);
    auto ov20_input_u = std::vector<uint8_t>(input_yuv_iter, input_yuv_iter + shape_size(input_u_shape));

    input_yuv_iter += shape_size(input_u_shape);
    auto ov20_input_v = std::vector<uint8_t>(input_yuv_iter, input_yuv_iter + shape_size(input_v_shape));

    ref_out_data.emplace_back(outType, input_y_shape, ov20_input_y.data());

    // Build model and set inputs
    function = build_test_model(inType, input_y_shape);
    set_test_model_color_conversion(ColorFormat::I420_THREE_PLANES, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), input_y_shape, ov20_input_y.data()});
    inputs.emplace(params.at(1), ov::Tensor{params.at(1)->get_element_type(), input_u_shape, ov20_input_u.data()});
    inputs.emplace(params.at(2), ov::Tensor{params.at(2)->get_element_type(), input_v_shape, ov20_input_v.data()});

    run();
}

TEST_P(PreprocessingYUV2GreyTest, convert_single_plane_nv12_use_opencv) {
    // Test various possible r/g/b values within dimensions
    const auto input_yuv_shape = Shape{1, get_full_height() * 3 / 2, width, 1};
    const auto input_y_shape = Shape{1, get_full_height(), width, 1};
    auto ov20_input_yuv = ov::test::utils::color_test_image(height, width, b_step, ColorFormat::NV12_SINGLE_PLANE);
    auto ov20_input_y =
        std::vector<uint8_t>(ov20_input_yuv.begin(), ov20_input_yuv.begin() + shape_size(input_y_shape));

    ref_out_data.emplace_back(outType, input_y_shape, ov20_input_y.data());

    // Build model and set inputs
    function = build_test_model(inType, input_y_shape);
    set_test_model_color_conversion(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), input_yuv_shape, ov20_input_yuv.data()});

    run();
}

TEST_P(PreprocessingYUV2GreyTest, convert_two_plane_nv12_use_opencv) {
    // Test various possible r/g/b values within dimensions
    const auto input_y_shape = Shape{1, get_full_height(), width, 1};
    const auto input_uv_shape = Shape{1, get_full_height() / 2, width / 2, 2};
    auto ov20_input_yuv = ov::test::utils::color_test_image(height, width, b_step, ColorFormat::NV12_TWO_PLANES);

    auto input_yuv_iter = ov20_input_yuv.begin();
    auto ov20_input_y = std::vector<uint8_t>(input_yuv_iter, input_yuv_iter + shape_size(input_y_shape));
    input_yuv_iter += shape_size(input_y_shape);

    auto ov20_input_uv = std::vector<uint8_t>(input_yuv_iter, input_yuv_iter + shape_size(input_uv_shape));
    input_yuv_iter += shape_size(input_uv_shape);

    ref_out_data.emplace_back(outType, input_y_shape, ov20_input_y.data());

    // Build model and set inputs
    function = build_test_model(inType, input_y_shape);
    set_test_model_color_conversion(ColorFormat::NV12_TWO_PLANES, ColorFormat::GRAY);

    const auto& params = function->get_parameters();
    inputs.emplace(params.at(0), ov::Tensor{params.at(0)->get_element_type(), input_y_shape, ov20_input_y.data()});
    inputs.emplace(params.at(1), ov::Tensor{params.at(1)->get_element_type(), input_uv_shape, ov20_input_uv.data()});

    run();
}
}  // namespace preprocess
}  // namespace ov
