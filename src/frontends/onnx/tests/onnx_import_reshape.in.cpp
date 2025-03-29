// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reshape_reduced_dims) {
    auto model = convert_model("reshape_reduced_dims.onnx");

    // input data shape (2, 3, 4)
    auto input = ov::test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                              {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (2, 12)
    auto expected_output = ov::test::NDArray<float, 2>({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                                        {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}})
                               .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{2, 12}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reshape_reordered_dims) {
    auto model = convert_model("reshape_reordered_dims.onnx");

    // input data shape (2, 3, 4)
    auto input = ov::test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                              {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (4, 2, 3)
    auto expected_output = ov::test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}},
                                                        {{6, 7, 8}, {9, 10, 11}},
                                                        {{12, 13, 14}, {15, 16, 17}},
                                                        {{18, 19, 20}, {21, 22, 23}}})
                               .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{4, 2, 3}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reshape_extended_dims) {
    auto model = convert_model("reshape_extended_dims.onnx");

    // input data shape (2, 3, 4)
    auto input = ov::test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                              {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (3, 2, 2, 2)
    auto expected_output = ov::test::NDArray<float, 4>({{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}},
                                                        {{{8, 9}, {10, 11}}, {{12, 13}, {14, 15}}},
                                                        {{{16, 17}, {18, 19}}, {{20, 21}, {22, 23}}}})
                               .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{3, 2, 2, 2}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reshape_single_dim) {
    auto model = convert_model("reshape_single_dim.onnx");

    // input data shape (2, 3, 4)
    auto input = ov::test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                              {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (24, )
    auto expected_output = ov::test::NDArray<float, 1>(
                               {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23})
                               .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{24}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reshape_negative_dim) {
    // the model contains the target shape in the initializers: [2, -1, 2]
    const auto model = convert_model("reshape_negative_dim.onnx");

    // 2x3x4
    auto input = ov::test::NDArray<float, 3>({{{0.5488135f, 0.71518934f, 0.60276335f, 0.5448832f},
                                               {0.4236548f, 0.6458941f, 0.4375872f, 0.891773f},
                                               {0.96366274f, 0.3834415f, 0.79172504f, 0.5288949f}},

                                              {{0.56804454f, 0.92559665f, 0.07103606f, 0.0871293f},
                                               {0.0202184f, 0.83261985f, 0.77815676f, 0.87001216f},
                                               {0.9786183f, 0.7991586f, 0.46147937f, 0.7805292f}}})
                     .get_vector();

    // 2x6x2
    auto expected_output = ov::test::NDArray<float, 3>({{{0.5488135f, 0.71518934f},
                                                         {0.60276335f, 0.5448832f},
                                                         {0.4236548f, 0.6458941f},
                                                         {0.4375872f, 0.891773f},
                                                         {0.96366274f, 0.3834415f},
                                                         {0.79172504f, 0.5288949f}},

                                                        {{0.56804454f, 0.92559665f},
                                                         {0.07103606f, 0.0871293f},
                                                         {0.0202184f, 0.83261985f},
                                                         {0.77815676f, 0.87001216f},
                                                         {0.9786183f, 0.7991586f},
                                                         {0.46147937f, 0.7805292f}}})
                               .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{2, 6, 2}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reshape_negative_with_zero_dim) {
    auto model = convert_model("reshape_negative_with_zero_dims.onnx");

    // input data shape (2, 3, 4)
    auto input = ov::test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                              {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (2, 6, 2)
    auto expected_output = ov::test::NDArray<float, 3>({{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
                                                        {{12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}}})
                               .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{2, 6, 2}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reshape_output_shape_as_input) {
    auto model = convert_model("reshape_output_shape_as_input.onnx");

    // input data shape (2, 3, 4)
    auto input = ov::test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                              {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (2, 6, 2)
    auto expected_output = ov::test::NDArray<float, 3>({{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
                                                        {{12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}}})
                               .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{2, 6, 2}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_depth_to_space) {
    auto model = convert_model("depth_to_space.onnx");

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0.f);

    std::vector<float> expected_output{0.f,  8.f,  1.f,  9.f,  16.f, 24.f, 17.f, 25.f, 2.f,  10.f, 3.f,
                                       11.f, 18.f, 26.f, 19.f, 27.f, 4.f,  12.f, 5.f,  13.f, 20.f, 28.f,
                                       21.f, 29.f, 6.f,  14.f, 7.f,  15.f, 22.f, 30.f, 23.f, 31.f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_v1) {
    auto model = convert_model("depth_to_space_v1.onnx");

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0.f);

    std::vector<float> expected_output{0.f,  8.f,  1.f,  9.f,  16.f, 24.f, 17.f, 25.f, 2.f,  10.f, 3.f,
                                       11.f, 18.f, 26.f, 19.f, 27.f, 4.f,  12.f, 5.f,  13.f, 20.f, 28.f,
                                       21.f, 29.f, 6.f,  14.f, 7.f,  15.f, 22.f, 30.f, 23.f, 31.f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_crd) {
    auto model = convert_model("depth_to_space_crd.onnx");

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0.f);

    std::vector<float> expected_output{0.f,  4.f,  1.f,  5.f,  8.f,  12.f, 9.f,  13.f, 2.f,  6.f,  3.f,
                                       7.f,  10.f, 14.f, 11.f, 15.f, 16.f, 20.f, 17.f, 21.f, 24.f, 28.f,
                                       25.f, 29.f, 18.f, 22.f, 19.f, 23.f, 26.f, 30.f, 27.f, 31.f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_bad_blocksize) {
    // This model fails to import since the depth channel length must be a multiple of the
    // `blocksize` attribute value.
    EXPECT_THROW(convert_model("depth_to_space_bad_blocksize.onnx"), std::runtime_error);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_no_blocksize) {
    // This model fails to import since it lacks of required parameter `blocksize`.
    EXPECT_THROW(convert_model("depth_to_space_no_blocksize.onnx"), std::runtime_error);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_bad_mode) {
    try {
        convert_model("depth_to_space_bad_mode.onnx");
        FAIL() << "The onnx_importer did not throw for unknown mode to DepthToSpace op";
    } catch (const ov::Exception& e) {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("only 'DCR' and 'CRD' modes are supported"), std::string::npos);
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_bad_input_shape) {
    try {
        convert_model("depth_to_space_bad_input_shape.onnx");
        FAIL() << "The onnx_importer did not throw for invalid input shape to DepthToSpace op";
    } catch (const ov::Exception& e) {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("Input must be 4-dimensional"), std::string::npos);
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_space_to_depth) {
    auto model = convert_model("space_to_depth.onnx");

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0.f);

    std::vector<float> expected_output{
        0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
        4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f, 5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
    };

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_space_to_depth_invalid_input_shape) {
    try {
        convert_model("space_to_depth_invalid_input_shape.onnx");
        FAIL() << "Expected ov::Exception, but no exception was thrown";
    } catch (const ov::Exception& e) {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("Input must be 4-dimensional"), std::string::npos)
            << "Could not find \"Input must be 4-dimensional\" in exception message: \"" << msg << "\"";
    } catch (...) {
        FAIL() << "Expected ov::Exception, got another type of exception";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_space_to_depth_bad_blocksize) {
    // This model fails to import since the depth channel length must be a multiple of the
    // `blocksize` attribute value.
    EXPECT_THROW(convert_model("space_to_depth_bad_blocksize.onnx"), std::runtime_error);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_space_to_depth_no_blocksize) {
    // This model fails to import since it lacks of required `blocksize` attribute.
    EXPECT_THROW(convert_model("space_to_depth_no_blocksize.onnx"), std::runtime_error);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_squeeze) {
    auto model = convert_model("squeeze.onnx");

    // {1, 4, 1, 1, 2}
    auto input = ov::test::NDArray<float, 5>({{{{{1.0f, 2.0f}}}, {{{3.0f, 4.0f}}}, {{{5.0f, 6.0f}}}, {{{7.0f, 8.0f}}}}})
                     .get_vector();

    // {4, 2}
    auto expected_output =
        ov::test::NDArray<float, 2>({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(Shape{1, 4, 1, 1, 2}, input);
    test_case.add_expected_output(Shape{4, 2}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_squeeze_empty_axes_attribute) {
    auto model = convert_model("squeeze_empty_axes_attribute.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input<float>(Shape{1, 4, 1, 1, 2}, data);
    test_case.add_expected_output<float>(Shape{4, 2}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_squeeze_opset13_no_axes) {
    const auto model = convert_model("squeeze_opset13_no_axes.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input<float>(Shape{1, 4, 1, 1, 2}, data);
    test_case.add_expected_output<float>(Shape{4, 2}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze) {
    auto model = convert_model("unsqueeze.onnx");

    auto input = ov::test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();

    auto expected_output =
        ov::test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_negative_axes) {
    auto model = convert_model("unsqueeze_negative_axes.onnx");

    auto input = ov::test::NDArray<float, 4>({{{{-1.8427763f, -1.0467733f, 0.50550157f, 1.4897262f, 0.33057404f}},
                                               {{1.9244908f, -0.3804572f, 0.76275414f, -0.8183123f, 0.93889356f}},
                                               {{-0.05270234f, 0.7113202f, -0.45783648f, -1.3378475f, 0.26926285f}}}})
                     .get_vector();

    auto expected_output =
        ov::test::NDArray<float, 5>({{{{{-1.8427763f, -1.0467733f, 0.50550157f, 1.4897262f, 0.33057404f}}},
                                      {{{1.9244908f, -0.3804572f, 0.76275414f, -0.8183123f, 0.93889356f}}},
                                      {{{-0.05270234f, 0.7113202f, -0.45783648f, -1.3378475f, 0.26926285f}}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_concat) {
    auto model = convert_model("concat.onnx");

    Inputs inputs;

    inputs.emplace_back(ov::test::NDArray<float, 1>({1, 2}).get_vector());
    inputs.emplace_back(ov::test::NDArray<float, 1>({3, 4}).get_vector());

    auto expected_output = ov::test::NDArray<float, 1>({1, 2, 3, 4}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_concat_negative_axis) {
    auto model = convert_model("concat_negative_axis.onnx");

    Inputs inputs;

    inputs.emplace_back(ov::test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    inputs.emplace_back(ov::test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto expected_output = ov::test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}, {7, 8}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_split_equal_parts_default) {
    auto model = convert_model("split_equal_parts_default.onnx");

    Inputs inputs{{1, 2, 3, 4, 5, 6}};
    Outputs expected_outputs{{1, 2}, {3, 4}, {5, 6}};
    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);

    for (std::size_t i = 0; i < expected_outputs.size(); ++i) {
        test_case.add_expected_output(expected_outputs[i]);
    }
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_split_equal_parts_2d) {
    // Split into 2 equal parts along axis=1
    auto model = convert_model("split_equal_parts_2d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>({0, 1, 2, 6, 7, 8});
    test_case.add_expected_output<float>({3, 4, 5, 9, 10, 11});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_split_variable_parts_2d) {
    // Split into variable parts {2, 4} along axis=1
    auto model = convert_model("split_variable_parts_2d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>({0, 1, 6, 7});
    test_case.add_expected_output<float>({2, 3, 4, 5, 8, 9, 10, 11});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_expand_static_shape) {
    auto model = convert_model("expand_static_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // input data shape (3,1)
    test_case.add_input(std::vector<float>{1, 2, 3});

    test_case.add_expected_output<float>(Shape{2, 3, 6}, {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                                                          1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3});

    test_case.run();
}
