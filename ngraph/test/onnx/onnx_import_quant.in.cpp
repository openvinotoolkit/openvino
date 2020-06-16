//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using TestEngine_t = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);
    test_case.add_input(std::vector<float>{32.25f, 48.34f, 50.f, 83.f});
    test_case.add_input(std::vector<float>{0.5f});

    test_case.add_expected_output(std::vector<std::uint8_t>{64, 97, 100, 166});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear_zero_point)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear_zero_point.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);
    test_case.add_input(std::vector<float>{0.f, 2.f, 3.f, 1000.f, -254.f, -1000.f}); // x
    test_case.add_input(std::vector<float>{2.0f});                                   // y_scale
    test_case.add_input(std::vector<std::uint8_t>{128});                             // y_zero_point

    test_case.add_expected_output<std::uint8_t>(
        {6}, std::vector<std::uint8_t>{128, 129, 130, 255, 1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear_axis_zero)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear_axis_zero.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);
    test_case.add_input(std::vector<float>{
        0.f, 2.f, 3.f, 1000.f, 0.f, 2.f, 3.f, 1000.f, 0.f, 2.f, 3.f, 1000.f}); // x
    test_case.add_input(std::vector<float>{1.f, 2.f, 4.f});                    // y_scale
    test_case.add_input(std::vector<std::uint8_t>{0, 0, 0});                   // y_zero_point

    //  std::vector<std::uint8_t>{0, 2, 3, 255, 0, 1, 2, 255, 0, 1, 1, 250}}; <- bad expected output
    //                                                                           given HALF_TO_EVEN
    //                                                                           round mode
    test_case.add_expected_output<std::uint8_t>(
        {3, 4}, std::vector<std::uint8_t>{0, 2, 3, 255, 0, 1, 2, 255, 0, 0, 1, 250});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear_axis_negative)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear_axis_negative.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);
    test_case.add_input(std::vector<float>{
        0.f, 2.f, 3.f, 1000.f, 0.f, 2.f, 3.f, 1000.f, 0.f, 2.f, 3.f, 1000.f}); // x
    test_case.add_input(std::vector<float>{1.f, 2.f, 4.f});                    // y_scale
    test_case.add_input(std::vector<std::uint8_t>{0, 0, 0});                   // y_zero_point

    //  std::vector<std::uint8_t>{0, 2, 3, 255, 0, 1, 2, 255, 0, 1, 1, 250}}; <- bad expected output
    //                                                                           given HALF_TO_EVEN
    //                                                                           round mode
    test_case.add_expected_output<std::uint8_t>(
        {3, 4}, std::vector<std::uint8_t>{0, 2, 3, 255, 0, 1, 2, 255, 0, 0, 1, 250});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequant_lin.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);
    test_case.add_input(std::vector<std::uint8_t>{19, 210, 21, 10});

    test_case.add_expected_output(std::vector<float>{76.f, 840.f, 84.f, 40.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_scalar_zero_scale_uint8)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_0.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);
    test_case.add_input(std::vector<uint8_t>{0, 3, 128, 255}); // x
    test_case.add_input(std::vector<float>{2.0f});             // scale
    test_case.add_input(std::vector<uint8_t>{128});            // zero_point

    test_case.add_expected_output<float>({4}, std::vector<float>{-256.0f, -250.0f, 0.0f, 254.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_scalar_zero_scale_int8)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_1.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<int8_t>{-30, -3, 100, 127}); // x
    test_case.add_input(std::vector<float>{2.0f});               // scale
    test_case.add_input(std::vector<int8_t>{-10});               // zero_point

    test_case.add_expected_output<float>({4}, std::vector<float>{-40.0f, 14.0f, 220.0f, 274.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_1d_zero_scale_uint8)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_2.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 10, 20, 30}); // x
    test_case.add_input(std::vector<float>{1.0f, 2.0f, 4.0f});                        // scale
    test_case.add_input(std::vector<uint8_t>{0, 0, 0});                               // zero_point

    test_case.add_expected_output<float>(
        {3, 4},
        std::vector<float>{
            0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 4.0f, 6.0f, 0.0f, 40.0f, 80.0f, 120.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_1d_zero_scale_int8)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_3.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<int8_t>{0, 1, 2, 3, 0, 2, 4, 6, 0, 10, 20, 30}); // x
    test_case.add_input(std::vector<float>{1.0f, 2.0f, 4.0f, 8.0f});                 // scale
    test_case.add_input(std::vector<int8_t>{0, -10, -20, -30});                      // zero_point

    test_case.add_expected_output<float>(
        {3, 4},
        std::vector<float>{
            0.0f, 22.0f, 88.0f, 264.0f, 0.0f, 24.0f, 96.0f, 288.0f, 0.0f, 40.0f, 160.0f, 480.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_1d_zero_scale_int8_4d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_4.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{7, 9, 10, 10, 5, 8, 9, 1, 8, 6, 7, 9, 10, 0, 7, 10, 8,
                                             2, 6, 0,  5,  9, 8, 1, 2, 7, 5, 3, 2, 4,  1, 3, 8,  7,
                                             4, 8, 10, 1,  5, 5, 7, 7, 0, 2, 4, 4, 0,  5}); // x
    test_case.add_input(std::vector<float>{1.0f, 10.0f, 7.0f});                             // scale
    test_case.add_input(std::vector<uint8_t>{10, 2, 1}); // zero_point

    test_case.add_expected_output<float>(
        {2, 3, 2, 4},
        std::vector<float>{-3.0f, -1.0f, 0.0f,  0.0f,   -5.0f, -2.0f, -1.0f, -9.0f,  60.0f, 40.0f,
                           50.0f, 70.0f, 80.0f, -20.0f, 50.0f, 80.0f, 49.0f, 7.0f,   35.0f, -7.0f,
                           28.0f, 56.0f, 49.0f, 0.0f,   -8.0f, -3.0f, -5.0f, -7.0f,  -8.0f, -6.0f,
                           -9.0f, -7.0f, 60.0f, 50.0f,  20.0f, 60.0f, 80.0f, -10.0f, 30.0f, 30.0f,
                           42.0f, 42.0f, -7.0f, 7.0f,   21.0f, 21.0f, -7.0f, 28.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_1d_zero_scale_uint8_negative_axis)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_5.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 10, 20, 30}); // x
    test_case.add_input(std::vector<float>{1.0f, 2.0f, 4.0f});                        // scale
    test_case.add_input(std::vector<uint8_t>{0, 0, 0});                               // zero_point

    test_case.add_expected_output<float>(
        {3, 4},
        std::vector<float>{
            0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 4.0f, 6.0f, 0.0f, 40.0f, 80.0f, 120.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quant_conv_linear)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quant_conv_lin.prototxt"));

    std::vector<std::vector<std::uint8_t>> inputs;
    inputs.emplace_back(std::vector<std::uint8_t>{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81});

    std::vector<std::vector<std::int8_t>> expected_output{std::vector<std::int8_t>{
        2,  3,  3,  3,  4,  4,  4,  5,  2,  4,  6,  7,  8,  8,  9,  9,  10, 3,  8,  11, 12,
        13, 13, 14, 14, 15, 5,  11, 16, 17, 18, 18, 19, 19, 20, 7,  14, 22, 22, 23, 23, 24,
        24, 25, 8,  18, 27, 27, 28, 28, 29, 29, 30, 10, 21, 32, 32, 33, 33, 34, 34, 35, 12,
        24, 37, 37, 38, 38, 39, 40, 40, 13, 17, 26, 27, 27, 27, 28, 28, 28, 9}};

    std::vector<std::vector<std::int8_t>> outputs{
        execute<std::uint8_t, std::int8_t>(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quant_conv_linear_2d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/qlinear_conv_2d.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input_from_file<uint8_t>(TEST_FILES, "onnx/qlinearconv2d/x.bin");
    test_case.add_input(std::vector<float>{0.00369204697199166f}); // x_scale
    test_case.add_input(std::vector<uint8_t>{132});                // x_zero_point
    test_case.add_input(std::vector<uint8_t>{0});                  // w
    test_case.add_input(std::vector<float>{0.00172794575337321f}); // w_scale
    test_case.add_input(std::vector<uint8_t>{255});                // w_zero_point
    test_case.add_input(std::vector<float>{0.00162681262008846f}); // y_scale
    test_case.add_input(std::vector<uint8_t>{123});                // y_zero_point

    test_case.add_expected_output_from_file<uint8_t>(
        {1, 1, 7, 7}, TEST_FILES, "onnx/qlinearconv2d/y.bin");
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quant_conv_linear_3d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/qlinear_conv_3d.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input_from_file<uint8_t>(TEST_FILES, "onnx/qlinearconv3d/x.bin");
    test_case.add_input(std::vector<float>{0.00389225385151803f}); // x_scale
    test_case.add_input(std::vector<uint8_t>{127});                // x_zero_point
    test_case.add_input(std::vector<uint8_t>{255});                // w
    test_case.add_input(std::vector<float>{0.00128723995294422f}); // w_scale
    test_case.add_input(std::vector<uint8_t>{0});                  // w_zero_point
    test_case.add_input(std::vector<float>{0.0011764180380851f});  // y_scale
    test_case.add_input(std::vector<uint8_t>{128});                // y_zero_point

    test_case.add_expected_output_from_file<uint8_t>(
        {1, 1, 4, 4, 4}, TEST_FILES, "onnx/qlinearconv3d/y.bin");
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_qlinear_matmul)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/qlinear_matmul.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{208, 236, 0, 238, 3, 214, 255, 29}); // T1
    test_case.add_input(std::vector<float>{0.0066f});                             // a_scale
    test_case.add_input(std::vector<uint8_t>{113});                               // a_zero_point
    test_case.add_input(
        std::vector<uint8_t>{152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247}); // T2
    test_case.add_input(std::vector<float>{0.00705f});                                // b_scale
    test_case.add_input(std::vector<uint8_t>{114});   // b_zero_point
    test_case.add_input(std::vector<float>{0.0107f}); // y_scale
    test_case.add_input(std::vector<uint8_t>{118});   // y_zero_point

    test_case.add_expected_output({2, 3}, std::vector<uint8_t>{168, 115, 255, 1, 66, 151}); // T3
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_qlinear_matmul_3d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/qlinear_matmul_3d.prototxt"));

    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{
        208, 236, 0, 238, 3, 214, 255, 29, 208, 236, 0, 238, 3, 214, 255, 29}); // T1
    test_case.add_input(std::vector<float>{0.0066f});                           // a_scale
    test_case.add_input(std::vector<uint8_t>{113});                             // a_zero_point
    test_case.add_input(std::vector<uint8_t>{152, 51,  244, 60,  26,  255, 0,   127,
                                             246, 127, 254, 247, 152, 51,  244, 60,
                                             26,  255, 0,   127, 246, 127, 254, 247}); // T2
    test_case.add_input(std::vector<float>{0.00705f});                                 // b_scale
    test_case.add_input(std::vector<uint8_t>{114});   // b_zero_point
    test_case.add_input(std::vector<float>{0.0107f}); // y_scale
    test_case.add_input(std::vector<uint8_t>{118});   // y_zero_point

    test_case.add_expected_output(
        {2, 2, 3},
        std::vector<uint8_t>{168, 115, 255, 1, 66, 151, 168, 115, 255, 1, 66, 151}); // T3
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{2, 3, 4, 5, 6, 7, 8, 9, 10}); // x
    test_case.add_input(std::vector<uint8_t>{1, 1, 1, 1});                 // w
    test_case.add_input(std::vector<uint8_t>{1});                          // x_zero_point

    test_case.add_expected_output({1, 1, 2, 2}, std::vector<uint8_t>{12, 16, 24, 28}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer_zero_point_zero)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9}); // x
    test_case.add_input(std::vector<uint8_t>{1, 1, 1, 1});                // w
    test_case.add_input(std::vector<uint8_t>{0});                         // x_zero_point

    test_case.add_expected_output({1, 1, 2, 2}, std::vector<uint8_t>{12, 16, 24, 28}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer_no_zero_point)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer_no_zero_point.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9}); // x
    test_case.add_input(std::vector<uint8_t>{1, 1, 1, 1});                // w

    test_case.add_expected_output({1, 1, 2, 2}, std::vector<uint8_t>{12, 16, 24, 28}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer_pads)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer_pads.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{2, 3, 4, 5, 6, 7, 8, 9, 10}); // x
    test_case.add_input(std::vector<uint8_t>{1, 1, 1, 1});                 // w
    test_case.add_input(std::vector<uint8_t>{1});                          // x_zero_point

    test_case.add_expected_output(
        {1, 1, 4, 4},
        std::vector<uint8_t>{1, 3, 5, 3, 5, 12, 16, 9, 11, 24, 28, 15, 7, 15, 17, 9}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0}); // a
    test_case.add_input(std::vector<uint8_t>{1, 4, 2, 5, 3, 6});                     // b
    test_case.add_input(std::vector<uint8_t>{12});                                   // a_zero_point
    test_case.add_input(std::vector<uint8_t>{0});                                    // b_zero_point

    test_case.add_expected_output(
        {4, 2}, std::vector<int32_t>{-38, -83, -44, -98, -50, -113, -56, -128}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_zero_point_zero)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0}); // a
    test_case.add_input(std::vector<uint8_t>{1, 4, 2, 5, 3, 6});                     // b
    test_case.add_input(std::vector<uint8_t>{0});                                    // a_zero_point
    test_case.add_input(std::vector<uint8_t>{0});                                    // b_zero_point

    test_case.add_expected_output({4, 2},
                                  std::vector<int32_t>{34, 97, 28, 82, 22, 67, 16, 52}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_no_zero_point)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_no_zero_point.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0}); // a
    test_case.add_input(std::vector<uint8_t>{1, 4, 2, 5, 3, 6});                     // b

    test_case.add_expected_output({4, 2},
                                  std::vector<int32_t>{34, 97, 28, 82, 22, 67, 16, 52}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_scalar)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_scalar.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{11}); // a
    test_case.add_input(std::vector<uint8_t>{13}); // b
    test_case.add_input(std::vector<uint8_t>{12}); // a_zero_point
    test_case.add_input(std::vector<uint8_t>{12}); // b_zero_point

    test_case.add_expected_output({1, 1}, std::vector<int32_t>{-1}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_4d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_4d.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}); // a
    test_case.add_input(std::vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}); // b
    test_case.add_input(std::vector<uint8_t>{0}); // a_zero_point
    test_case.add_input(std::vector<uint8_t>{0}); // b_zero_point

    test_case.add_expected_output<int32_t>(Shape{1, 2, 3, 3},
                                           {42,
                                            48,
                                            54,
                                            114,
                                            136,
                                            158,
                                            186,
                                            224,
                                            262,
                                            906,
                                            960,
                                            1014,
                                            1170,
                                            1240,
                                            1310,
                                            1434,
                                            1520,
                                            1606}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_4d_zero_point)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_4d.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}); // a
    test_case.add_input(std::vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}); // b
    test_case.add_input(std::vector<uint8_t>{1}); // a_zero_point
    test_case.add_input(std::vector<uint8_t>{1}); // b_zero_point

    test_case.add_expected_output<int32_t>(Shape{1, 2, 3, 3},
                                           {22,
                                            24,
                                            26,
                                            78,
                                            96,
                                            114,
                                            134,
                                            168,
                                            202,
                                            790,
                                            840,
                                            890,
                                            1038,
                                            1104,
                                            1170,
                                            1286,
                                            1368,
                                            1450}); // y
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_4d_no_zero_point)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_4d_no_zero_point.prototxt"));
    auto test_case = test::TestCase<TestEngine_t>(function);

    test_case.add_input(std::vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}); // a
    test_case.add_input(std::vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}); // b

    test_case.add_expected_output<int32_t>(Shape{1, 2, 3, 3},
                                           {42,
                                            48,
                                            54,
                                            114,
                                            136,
                                            158,
                                            186,
                                            224,
                                            262,
                                            906,
                                            960,
                                            1014,
                                            1170,
                                            1240,
                                            1310,
                                            1434,
                                            1520,
                                            1606}); // y
    test_case.run();
}
