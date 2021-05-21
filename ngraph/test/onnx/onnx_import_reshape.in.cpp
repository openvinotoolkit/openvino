// Copyright (C) 2018-2021 Intel Corporation
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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "onnx_import/onnx.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reshape_reduced_dims)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_reduced_dims.prototxt"));

    // input data shape (2, 3, 4)
    auto input = test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                          {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (2, 12)
    auto expected_output =
        test::NDArray<float, 2>({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                 {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}})
            .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{2, 12}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reshape_reordered_dims)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_reordered_dims.prototxt"));

    // input data shape (2, 3, 4)
    auto input = test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                          {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (4, 2, 3)
    auto expected_output = test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}},
                                                    {{6, 7, 8}, {9, 10, 11}},
                                                    {{12, 13, 14}, {15, 16, 17}},
                                                    {{18, 19, 20}, {21, 22, 23}}})
                               .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{4, 2, 3}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reshape_extended_dims)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_extended_dims.prototxt"));

    // input data shape (2, 3, 4)
    auto input = test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                          {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (3, 2, 2, 2)
    auto expected_output = test::NDArray<float, 4>({{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}},
                                                    {{{8, 9}, {10, 11}}, {{12, 13}, {14, 15}}},
                                                    {{{16, 17}, {18, 19}}, {{20, 21}, {22, 23}}}})
                               .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{3, 2, 2, 2}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reshape_single_dim)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_single_dim.prototxt"));

    // input data shape (2, 3, 4)
    auto input = test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                          {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (24, )
    auto expected_output = test::NDArray<float, 1>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23})
                               .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{24}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reshape_negative_dim)
{
    // the model contains the target shape in the initializers: [2, -1, 2]
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_negative_dim.prototxt"));

    // 2x3x4
    auto input = test::NDArray<float, 3>({{{0.5488135, 0.71518934, 0.60276335, 0.5448832},
                                           {0.4236548, 0.6458941, 0.4375872, 0.891773},
                                           {0.96366274, 0.3834415, 0.79172504, 0.5288949}},

                                          {{0.56804454, 0.92559665, 0.07103606, 0.0871293},
                                           {0.0202184, 0.83261985, 0.77815676, 0.87001216},
                                           {0.9786183, 0.7991586, 0.46147937, 0.7805292}}})
                     .get_vector();

    // 2x6x2
    auto expected_output = test::NDArray<float, 3>({{{0.5488135, 0.71518934},
                                                     {0.60276335, 0.5448832},
                                                     {0.4236548, 0.6458941},
                                                     {0.4375872, 0.891773},
                                                     {0.96366274, 0.3834415},
                                                     {0.79172504, 0.5288949}},

                                                    {{0.56804454, 0.92559665},
                                                     {0.07103606, 0.0871293},
                                                     {0.0202184, 0.83261985},
                                                     {0.77815676, 0.87001216},
                                                     {0.9786183, 0.7991586},
                                                     {0.46147937, 0.7805292}}})
                               .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{2, 6, 2}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reshape_negative_with_zero_dim)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_negative_with_zero_dims.prototxt"));

    // input data shape (2, 3, 4)
    auto input = test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                          {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (2, 6, 2)
    auto expected_output =
        test::NDArray<float, 3>({{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
                                 {{12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}}})
            .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{2, 6, 2}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_reshape_output_shape_as_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_output_shape_as_input.prototxt"));

    // input data shape (2, 3, 4)
    auto input = test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                          {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                     .get_vector();

    // output data shape (2, 6, 2)
    auto expected_output =
        test::NDArray<float, 3>({{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
                                 {{12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}}})
            .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(Shape{2, 3, 4}, input);
    test_case.add_expected_output(Shape{2, 6, 2}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_depth_to_space)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/depth_to_space.prototxt"));

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0);

    std::vector<float> expected_output{
        0.f, 8.f,  1.f, 9.f,  16.f, 24.f, 17.f, 25.f, 2.f, 10.f, 3.f, 11.f, 18.f, 26.f, 19.f, 27.f,
        4.f, 12.f, 5.f, 13.f, 20.f, 28.f, 21.f, 29.f, 6.f, 14.f, 7.f, 15.f, 22.f, 30.f, 23.f, 31.f};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_v1)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/depth_to_space_v1.prototxt"));

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0);

    std::vector<float> expected_output{
        0.f, 8.f,  1.f, 9.f,  16.f, 24.f, 17.f, 25.f, 2.f, 10.f, 3.f, 11.f, 18.f, 26.f, 19.f, 27.f,
        4.f, 12.f, 5.f, 13.f, 20.f, 28.f, 21.f, 29.f, 6.f, 14.f, 7.f, 15.f, 22.f, 30.f, 23.f, 31.f};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_crd)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/depth_to_space_crd.prototxt"));

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0);

    std::vector<float> expected_output{0.f,  4.f,  1.f,  5.f,  8.f,  12.f, 9.f,  13.f,
                                       2.f,  6.f,  3.f,  7.f,  10.f, 14.f, 11.f, 15.f,
                                       16.f, 20.f, 17.f, 21.f, 24.f, 28.f, 25.f, 29.f,
                                       18.f, 22.f, 19.f, 23.f, 26.f, 30.f, 27.f, 31.f};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_bad_blocksize)
{
    // This model fails to import since the depth channel length must be a multiple of the
    // `blocksize` attribute value.
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(
                     SERIALIZED_ZOO, "onnx/depth_to_space_bad_blocksize.prototxt")),
                 std::runtime_error);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_no_blocksize)
{
    // This model fails to import since it lacks of required parameter `blocksize`.
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(
                     SERIALIZED_ZOO, "onnx/depth_to_space_no_blocksize.prototxt")),
                 std::runtime_error);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_bad_mode)
{
    try
    {
        onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/depth_to_space_bad_mode.prototxt"));
        FAIL() << "The onnx_importer did not throw for unknown mode to DepthToSpace op";
    }
    catch (const ngraph::ngraph_error& e)
    {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("only 'DCR' and 'CRD' modes are supported"), std::string::npos);
    }
    catch (...)
    {
        FAIL() << "Expected ngraph_error exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_depth_to_space_bad_input_shape)
{
    try
    {
        onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/depth_to_space_bad_input_shape.prototxt"));
        FAIL() << "The onnx_importer did not throw for invalid input shape to DepthToSpace op";
    }
    catch (const ngraph::ngraph_error& e)
    {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("Input must be 4-dimensional"), std::string::npos);
    }
    catch (...)
    {
        FAIL() << "Expected ngraph_error exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_space_to_depth)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/space_to_depth.prototxt"));

    std::vector<float> input(32);
    std::iota(input.begin(), input.end(), 0);

    std::vector<float> expected_output{
        0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
        4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f, 5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
    };

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_space_to_depth_invalid_input_shape)
{
    try
    {
        onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO, "onnx/space_to_depth_invalid_input_shape.prototxt"));
        FAIL() << "Expected ngraph_error exception, but no exception was thrown";
    }
    catch (const ngraph::ngraph_error& e)
    {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("Input must be 4-dimensional"), std::string::npos)
            << "Could not find \"Input must be 4-dimensional\" in exception message: \"" << msg
            << "\"";
    }
    catch (...)
    {
        FAIL() << "Expected ngraph_error exception, got another type of exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_space_to_depth_bad_blocksize)
{
    // This model fails to import since the depth channel length must be a multiple of the
    // `blocksize` attribute value.
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(
                     SERIALIZED_ZOO, "onnx/space_to_depth_bad_blocksize.prototxt")),
                 std::runtime_error);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_space_to_depth_no_blocksize)
{
    // This model fails to import since it lacks of required `blocksize` attribute.
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(
                     SERIALIZED_ZOO, "onnx/space_to_depth_no_blocksize.prototxt")),
                 std::runtime_error);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_squeeze)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/squeeze.prototxt"));

    // {1, 4, 1, 1, 2}
    auto input = test::NDArray<float, 5>(
                     {{{{{1.0f, 2.0f}}}, {{{3.0f, 4.0f}}}, {{{5.0f, 6.0f}}}, {{{7.0f, 8.0f}}}}})
                     .get_vector();

    // {4, 2}
    auto expected_output =
        test::NDArray<float, 2>({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}})
            .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(Shape{1, 4, 1, 1, 2}, input);
    test_case.add_expected_output(Shape{4, 2}, expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_squeeze_opset13_no_axes)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/squeeze_opset13_no_axes.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input<float>(Shape{1, 4, 1, 1, 2}, data);
    test_case.add_expected_output<float>(Shape{4, 2}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unsqueeze)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/unsqueeze.prototxt"));

    auto input = test::NDArray<float, 3>(
                     {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();

    auto expected_output =
        test::NDArray<float, 4>(
            {{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_negative_axes)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/unsqueeze_negative_axes.prototxt"));

    auto input = test::NDArray<float, 4>(
                     {{{{-1.8427763f, -1.0467733f, 0.50550157f, 1.4897262f, 0.33057404f}},
                       {{1.9244908f, -0.3804572f, 0.76275414f, -0.8183123f, 0.93889356f}},
                       {{-0.05270234f, 0.7113202f, -0.45783648f, -1.3378475f, 0.26926285f}}}})
                     .get_vector();

    auto expected_output =
        test::NDArray<float, 5>(
            {{{{{-1.8427763f, -1.0467733f, 0.50550157f, 1.4897262f, 0.33057404f}}},
              {{{1.9244908f, -0.3804572f, 0.76275414f, -0.8183123f, 0.93889356f}}},
              {{{-0.05270234f, 0.7113202f, -0.45783648f, -1.3378475f, 0.26926285f}}}}})
            .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_concat)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/concat.prototxt"));

    Inputs inputs;

    inputs.emplace_back(test::NDArray<float, 1>({1, 2}).get_vector());
    inputs.emplace_back(test::NDArray<float, 1>({3, 4}).get_vector());

    auto expected_output = test::NDArray<float, 1>({1, 2, 3, 4}).get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_concat_negative_axis)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/concat_negative_axis.prototxt"));

    Inputs inputs;

    inputs.emplace_back(test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto expected_output = test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}, {7, 8}}).get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_split_equal_parts_default)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_default.prototxt"));

    Inputs inputs{{1, 2, 3, 4, 5, 6}};
    Outputs expected_outputs{{1, 2}, {3, 4}, {5, 6}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());

    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(outputs[i].size(), expected_outputs[i].size());
        EXPECT_TRUE(test::all_close_f(outputs[i], expected_outputs[i]));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_split_equal_parts_2d)
{
    // Split into 2 equal parts along axis=1
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_2d.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>({0, 1, 2, 6, 7, 8});
    test_case.add_expected_output<float>({3, 4, 5, 9, 10, 11});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_split_variable_parts_2d)
{
    // Split into variable parts {2, 4} along axis=1
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_variable_parts_2d.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>({0, 1, 6, 7});
    test_case.add_expected_output<float>({2, 3, 4, 5, 8, 9, 10, 11});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_expand_static_shape)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/expand_static_shape.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // input data shape (3,1)
    test_case.add_input(std::vector<float>{1, 2, 3});

    test_case.add_expected_output<float>(Shape{2, 3, 6},
                                         {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                                          1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3});

    test_case.run();
}
