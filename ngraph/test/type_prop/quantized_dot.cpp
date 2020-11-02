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
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, quantized_dot_8_bit_output)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                   input1,
                                                   1,
                                                   scale,
                                                   input0_zero_point,
                                                   scale,
                                                   input1_zero_point,
                                                   scale,
                                                   output_zero_point,
                                                   output_type,
                                                   axes,
                                                   axes,
                                                   axes);

    ASSERT_EQ(quant_dot->get_element_type(), output_type);
    ASSERT_EQ(quant_dot->get_shape(), output_shape);
}

TEST(type_prop, quantized_dot_32_bit_output)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type i32 = element::i32;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i32;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i32;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                   input1,
                                                   1,
                                                   scale,
                                                   input0_zero_point,
                                                   scale,
                                                   input1_zero_point,
                                                   scale,
                                                   output_zero_point,
                                                   output_type,
                                                   axes,
                                                   axes,
                                                   axes);

    ASSERT_EQ(quant_dot->get_element_type(), output_type);
    ASSERT_EQ(quant_dot->get_shape(), output_shape);
}

TEST(type_prop, quantized_dot_non_quantized_input0_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = f32;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use non-quantized input0 not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input0 element type (f32) must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_non_quantized_input1_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = f32;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use non-quantized input1 not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input1 element type (f32) must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_dyn_output_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = element::dynamic;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use dynamic output type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Output element type must not be dynamic");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_non_floating_point_scale_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = i8;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to non floating point scale not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Scale must be a floating point number");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_input0_zero_point_type_mismatch_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = i8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use zero point type different from input0 type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Input0 Zero point element type (i8) must match input0 element type (u8)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_input1_zero_point_type_mismatch_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = u8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use zero point type different from input1 type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Input1 Zero point element type (u8) must match input1 element type (i8)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_non_scalar_input0_zero_point_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{1, 2});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use non scalar input0 zero point not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input0 scale and input0 zero point shape must be same and 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_non_scalar_input1_zero_point_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{1, 2});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use non scalar input1 zero point not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input1 scale and input1 zero point shape must be same and 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_non_scalar_output_zero_point_fails)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{1, 2});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use non scalar output zero point not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Output scale and output zero point shape must be same and 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_non_empty_input0_axes)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       AxisSet{1},
                                                       axes,
                                                       axes);

        FAIL() << "Attempt to use non empty input0 axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input0, input1 and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_non_empty_input1_axes)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       AxisSet{1},
                                                       axes);

        FAIL() << "Attempt to use non empty input1 axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input0, input1 and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_dot_non_empty_output_axes)
{
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input0_type = u8;
    element::Type input1_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input0_zero_point_type = u8;
    element::Type input1_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 72};
    AxisSet axes{};

    auto input0 = make_shared<op::Parameter>(input0_type, Shape{64, 3});
    auto input1 = make_shared<op::Parameter>(input1_type, Shape{3, 72});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input0_zero_point = make_shared<op::Parameter>(input0_zero_point_type, Shape{});
    auto input1_zero_point = make_shared<op::Parameter>(input1_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_dot = make_shared<op::QuantizedDot>(input0,
                                                       input1,
                                                       1,
                                                       scale,
                                                       input0_zero_point,
                                                       scale,
                                                       input1_zero_point,
                                                       scale,
                                                       output_zero_point,
                                                       output_type,
                                                       axes,
                                                       axes,
                                                       AxisSet{1});

        FAIL() << "Attempt to use non empty output axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input0, input1 and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
