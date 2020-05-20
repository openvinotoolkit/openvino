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

using namespace std;
using namespace ngraph;

TEST(type_prop, quantized_conv_8_bit_output)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                            filter,
                                                            strides,
                                                            dilation,
                                                            padding_below,
                                                            padding_above,
                                                            dilation,
                                                            scale,
                                                            u8_zero_point,
                                                            scale,
                                                            i8_zero_point,
                                                            scale,
                                                            i8_zero_point,
                                                            output_type,
                                                            axes,
                                                            axes,
                                                            axes);

    ASSERT_EQ(quant_conv->get_element_type(), output_type);
    ASSERT_EQ(quant_conv->get_shape(), output_shape);
}

TEST(type_prop, quantized_conv_32_bit_output)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type i32 = element::i32;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i32;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                            filter,
                                                            strides,
                                                            dilation,
                                                            padding_below,
                                                            padding_above,
                                                            dilation,
                                                            scale,
                                                            u8_zero_point,
                                                            scale,
                                                            i8_zero_point,
                                                            scale,
                                                            i8_zero_point,
                                                            output_type,
                                                            axes,
                                                            axes,
                                                            axes);

    ASSERT_EQ(quant_conv->get_element_type(), output_type);
    ASSERT_EQ(quant_conv->get_shape(), output_shape);
}

TEST(type_prop, quantized_conv_non_quantized_input_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = f32;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                u8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non-quantized input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input element type (f32) must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_quantized_filter_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = f32;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                u8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non-quantized filter not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Filter element type (f32) must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_dyn_output_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = f32;
    element::Type output_type = element::dynamic;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                u8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                scale,
                                                                i8_zero_point,
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

TEST(type_prop, quantized_conv_non_floating_point_scale_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = i8;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                u8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non floating point scale not detected";
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

TEST(type_prop, quantized_conv_input_zero_point_type_mismatch_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = i8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use zero point type different from input type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), "Input Zero point element type (i8) must match input element type (u8)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_filter_zero_point_type_mismatch_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = u8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use zero point type different from filter type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Filter Zero point element type (u8) must match filter element type (i8)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_scalar_input_zero_point_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{1, 2});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non scalar input zero point not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input scale and input zero point shape must be same and 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_scalar_filter_zero_point_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{1, 2});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non scalar filter zero point not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Filter scale and filter zero point shape must be same and 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_scalar_output_zero_point_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{1, 2});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
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

TEST(type_prop, quantized_conv_non_empty_input_axes)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                AxisSet{1},
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non empty input axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input, filter and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_empty_filter_axes)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                AxisSet{1},
                                                                axes);
        FAIL() << "Attempt to use non empty filter axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input, filter and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_empty_output_axes)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
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
        EXPECT_HAS_SUBSTRING(error.what(), "Input, filter and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
