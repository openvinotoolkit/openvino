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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, quantize)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divide by scale                2  2  2  2  2  2  2  2  2  2  2   2
    // equals (rounded)               0  0  1  2  2  2  3  4  4  4  5   6
    // plus offset                    1  1  1  1  1  1  1  1  1  1  1   1
    // equals                         1  1  2  3  3  3  4  5  5  5  6   7

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {1});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7});
    // minus offset                   1  1  1  1  1  1  1  1  1  1  1  1
    // eqauls                         0  0  1  2  2  2  3  4  4  4  5  6
    // multiplied by scale            2  2  2  2  2  2  2  2  2  2  2  2
    // equals                         0  0  2  4  4  4  6  8  8  8 10 12

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_TRUE(test::all_close_f((vector<output_c_type>{0, 0, 2, 4, 4, 4, 6, 8, 8, 8, 10, 12}),
                                  read_vector<output_c_type>(y),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divide by scale                2  2  2  2  2  2  2  2  2  2  2   2
    // equals (rounded)               0  0  1  2  2  2  3  4  4  4  5   6
    // plus offset                    0  0  0  0  0  0  0  0  0  0  0   0
    // equals                         0  0  1  2  2  2  3  4  4  4  5   6

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {0});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6});
    // minus offset                   0  0  0  0  0  0  0  0  0  0  0  0
    // equals                         0  0  1  2  2  2  3  4  4  4  5  6
    // multiplied by scale            2  2  2  2  2  2  2  2  2  2  2  2
    // equals                         0  0  2  4  4  4  6  8  8  8 10 12

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_TRUE(test::all_close_f((vector<output_c_type>{0, 0, 2, 4, 4, 4, 6, 8, 8, 8, 10, 12}),
                                  read_vector<output_c_type>(y),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_axes)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape{4};
    AxisSet quantization_axes{0};

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2, 3, 4, 5});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {10, 20, 30, 40});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divided by scale               2  2  2  3  3  3  4  4  4  5  5   5
    // equals (rounded)               0  1  1  1  1  2  2  2  2  2  2   2
    // plus offset                   10 10 10 20 20 20 30 30 30 40 40  40
    // equals                        10 11 11 21 21 22 32 32 32 42 42  42

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{10, 11, 11, 21, 21, 22, 32, 32, 32, 42, 42, 42}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_axes)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape{4};
    AxisSet quantization_axes{0};

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2, 3, 4, 5});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {10, 20, 30, 40});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{10, 11, 11, 21, 21, 22, 32, 32, 32, 42, 42, 42});
    // minus offset                   10  10  10  20  20  20  30  30  30  40  40  40
    // equals                          0   1   1   1   1   2   2   2   2   2   2   2
    // multiplied by scale             2   2   2   3   3   3   4   4   4   5   5   5
    // equals                          0   2   2   3   3   6   8   8   8  10  10  10

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_TRUE(test::all_close_f((vector<output_c_type>{0, 2, 2, 3, 3, 6, 8, 8, 8, 10, 10, 10}),
                                  read_vector<output_c_type>(y),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0   0  1  -2  2  -2  3  -4  4  -4  5   -6
    // plus offset                    1   1  1   1  1   1  1   1  1   1  1    1
    // equals                         1   1  2  -1  3  -1  4  -3  5  -3  6   -5

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{1, 1, 2, -1, 3, -1, 4, -3, 5, -3, 6, -5}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i8;
    auto output_type = element::f32;

    typedef int8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {1});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{1, 1, 2, -1, 3, -1, 4, -3, 5, -3, 6, -5});
    // minus offset                   1  1  1   1  1   1  1   1  1   1  1   1
    // equals                         0  0  1  -2  2  -2  3  -4  4  -4  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0  0  2  -4  4  -4  6  -8  8  -8 10 -12

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_TRUE(
        test::all_close_f((vector<output_c_type>{0, 0, 2, -4, 4, -4, 6, -8, 8, -8, 10, -12}),
                          read_vector<output_c_type>(y),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int8_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0   0  1  -2  2  -2  3  -4  4  -4  5   -6
    // plus offset                    0   0  0   0  0   0  0   0  0   0  0    0
    // equals                         0   0  1  -2  2  -2  3  -4  4  -4  5   -6

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 1, -2, 2, -2, 3, -4, 4, -4, 5, -6}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int8_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i8;
    auto output_type = element::f32;

    typedef int8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {0});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 0, 1, -2, 2, -2, 3, -4, 4, -4, 5, -6});
    // minus offset                   0  0  0   0  0   0  0   0  0   0  0   0
    // equals                         0  0  1  -2  2  -2  3  -4  4  -4  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0  0  2  -4  4  -4  6  -8  8  -8 10 -12

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_TRUE(
        test::all_close_f((vector<output_c_type>{0, 0, 2, -4, 4, -4, 6, -8, 8, -8, 10, -12}),
                          read_vector<output_c_type>(y),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int32)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i32;

    typedef float input_c_type;
    typedef int32_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0   0  1  -2  2  -2  3  -4  4  -4  5   -6
    // plus offset                    1   1  1   1  1   1  1   1  1   1  1    1
    // equals                         1   1  2  -1  3  -1  4  -3  5  -3  6   -5

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{1, 1, 2, -1, 3, -1, 4, -3, 5, -3, 6, -5}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int32)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i32;
    auto output_type = element::f32;

    typedef int32_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {1});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{1, 1, 2, -1, 3, -1, 4, -3, 5, -3, 6, -5});
    // minus offset                   1  1  1   1  1   1  1   1  1   1  1   1
    // equals                         0  0  1  -2  2  -2  3  -4  4  -4  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0  0  2  -4  4  -4  6  -8  8  -8 10 -12

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_TRUE(
        test::all_close_f((vector<output_c_type>{0, 0, 2, -4, 4, -4, 6, -8, 8, -8, 10, -12}),
                          read_vector<output_c_type>(y),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int32_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i32;

    typedef float input_c_type;
    typedef int32_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0   0  1  -2  2  -2  3  -4  4  -4  5   -6
    // plus offset                    0   0  0   0  0   0  0   0  0   0  0    0
    // equals                         0   0  1  -2  2  -2  3  -4  4  -4  5   -6

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 1, -2, 2, -2, 3, -4, 4, -4, 5, -6}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int32_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i32;
    auto output_type = element::f32;

    typedef int32_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {0});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 0, 1, -2, 2, -2, 3, -4, 4, -4, 5, -6});
    // minus offset                   0  0  0   0  0   0  0   0  0   0  0   0
    // equals                         0  0  1  -2  2  -2  3  -4  4  -4  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0  0  2  -4  4  -4  6  -8  8  -8 10 -12

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_TRUE(
        test::all_close_f((vector<output_c_type>{0, 0, 2, -4, 4, -4, 6, -8, 8, -8, 10, -12}),
                          read_vector<output_c_type>(y),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_clamp_uint8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto max = std::numeric_limits<uint8_t>::max();

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {1.0 / (max + 1.0)});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, max, max, max, max, max, max, max, max, max, max, max}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_clamp_int8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto min = std::numeric_limits<int8_t>::min();
    auto max = std::numeric_limits<int8_t>::max();

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {1.0 / (max + 1.0)});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, min, max, min, max, min, max, min, max, min, max, min}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_clamp_int32)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f64;
    auto output_type = element::i32;

    // TODO: fails with input due to 32 bits
    typedef double input_c_type;
    typedef int32_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto min = std::numeric_limits<int32_t>::min();
    auto max = std::numeric_limits<int32_t>::max();

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {1.0 / (max + 1.0)});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, min, max, min, max, min, max, min, max, min, max, min}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_TOWARD_ZERO)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_ZERO;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   3  -2   -2   -3   3   3   4   -3   -3   -4

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 3, -2, -2, -3, 3, 3, 4, -3, -3, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_TOWARD_INFINITY)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   3   3  -2   -3   -3   3   4   4   -3   -4   -4

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 3, 3, -2, -3, -3, 3, 4, 4, -3, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_UPWARD)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_UPWARD;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   3   3  -2   -2   -3   3   4   4   -3   -3   -4

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 3, 3, -2, -2, -3, 3, 4, 4, -3, -3, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_DOWNWARD)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_DOWNWARD;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   3  -2   -3   -3   3   3   4   -3   -4   -4

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 3, -2, -3, -3, 3, 3, 4, -3, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_TOWARD_EVEN)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   3  -2   -2   -3   3   4   4   -3   -4   -4

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 3, -2, -2, -3, 3, 4, 4, -3, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_TOWARD_INFINITY)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_TOWARD_INFINITY;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize = make_shared<op::Quantize>(
        X,
        scale,
        offset,
        output_type,
        quantization_axes,
        static_cast<op::Quantize::RoundMode>(static_cast<int>(round_mode)));
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               3   3   3  -3   -3   -3   4   4   4   -4   -4   -4

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{3, 3, 3, -3, -3, -3, 4, 4, 4, -4, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_TOWARD_ZERO)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_TOWARD_ZERO;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize = make_shared<op::Quantize>(
        X,
        scale,
        offset,
        output_type,
        quantization_axes,
        static_cast<op::Quantize::RoundMode>(static_cast<int>(round_mode)));
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   2  -2   -2   -2   3   3   3   -3   -3   -3

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 2, -2, -2, -2, 3, 3, 3, -3, -3, -3}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_UP)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_UP;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               3   3   3  -2   -2   -2   4   4   4   -3   -3   -3

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{3, 3, 3, -2, -2, -2, 4, 4, 4, -3, -3, -3}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_DOWN)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_DOWN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   2  -3   -3   -3   3   3   3   -4   -4   -4

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 2, -3, -3, -3, 3, 3, 3, -4, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_dynamic_offset)
{
    Shape input_shape{4};
    Shape scale_offset_shape = {};
    AxisSet quantization_axes;

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = make_shared<op::Parameter>(output_type, scale_offset_shape);
    auto offset = make_shared<op::Parameter>(input_type, scale_offset_shape);
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X, scale, offset});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);
    auto Scale = backend->create_tensor(output_type, scale_offset_shape);
    auto Offset = backend->create_tensor(input_type, scale_offset_shape);

    copy_data(x, vector<input_c_type>{0, 3, 128, 255});
    copy_data(Scale, vector<output_c_type>{2});
    copy_data(Offset, vector<input_c_type>{128});

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x, Scale, Offset});
    EXPECT_TRUE(test::all_close_f((vector<output_c_type>{-256.0f, -250.0f, 0.0f, 254.0f}),
                                  read_vector<output_c_type>(y),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_dynamic_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape = {};
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = make_shared<op::Parameter>(input_type, scale_offset_shape);
    auto offset = make_shared<op::Parameter>(output_type, scale_offset_shape);
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X, scale, offset});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);
    auto Scale = backend->create_tensor(input_type, scale_offset_shape);
    auto Offset = backend->create_tensor(output_type, scale_offset_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divide by scale                2  2  2  2  2  2  2  2  2  2  2   2
    // equals (rounded)               0  0  1  2  2  2  3  4  4  4  5   6
    // plus offset                    1  1  1  1  1  1  1  1  1  1  1   1
    // equals                         1  1  2  3  3  3  4  5  5  5  6   7
    copy_data(Scale, vector<input_c_type>{2});
    copy_data(Offset, vector<output_c_type>{1});

    auto handle = backend->compile(f);
    handle->call_with_validate({y}, {x, Scale, Offset});
    EXPECT_EQ((vector<output_c_type>{1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7}),
              read_vector<output_c_type>(y));
}
