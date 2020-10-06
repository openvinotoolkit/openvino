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

#include "quantized_convolution.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/validation_util.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::QuantizedConvolution::type_info;

op::QuantizedConvolution::QuantizedConvolution(const Output<Node>& input,
                                               const Output<Node>& filters,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               const Output<Node>& input_scale,
                                               const Output<Node>& input_zero_point,
                                               const Output<Node>& filter_scale,
                                               const Output<Node>& filter_zero_point,
                                               const Output<Node>& output_scale,
                                               const Output<Node>& output_zero_point,
                                               const element::Type& output_type,
                                               const AxisSet& input_axes,
                                               const AxisSet& filter_axes,
                                               const AxisSet& output_axes)
    : Op({input,
          filters,
          input_scale,
          input_zero_point,
          filter_scale,
          filter_zero_point,
          output_scale,
          output_zero_point})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_output_type(output_type)
    , m_input_axes(input_axes)
    , m_filter_axes(filter_axes)
    , m_output_axes(output_axes)
{
    constructor_validate_and_infer_types();
}

void op::QuantizedConvolution::validate_and_infer_types()
{
    enum
    {
        INPUT,
        FILTER,
        INPUT_SCALE,
        INPUT_ZERO_POINT,
        FILTER_SCALE,
        FILTER_ZERO_POINT,
        OUTPUT_SCALE,
        OUTPUT_ZERO_POINT
    };

    NODE_VALIDATION_CHECK(
        this, m_output_type.is_static(), "Output element type must not be dynamic");

    NODE_VALIDATION_CHECK(this,
                          m_output_type.is_quantized(),
                          "Output element type (",
                          m_output_type,
                          ") must be a quantized type");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INPUT).is_quantized(),
                          "Input element type (",
                          get_input_element_type(INPUT),
                          ") must be a quantized type");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(FILTER).is_quantized(),
                          "Filter element type (",
                          get_input_element_type(FILTER),
                          ") must be a quantized type");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INPUT_SCALE).is_real() ||
                              get_input_element_type(INPUT_SCALE).is_dynamic() ||
                              get_input_element_type(FILTER_SCALE).is_real() ||
                              get_input_element_type(FILTER_SCALE).is_dynamic() ||
                              get_input_element_type(OUTPUT_SCALE).is_real() ||
                              get_input_element_type(OUTPUT_SCALE).is_dynamic(),
                          "Scale must be a floating point number");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(0).compatible(get_input_element_type(INPUT_ZERO_POINT)),
        "Input Zero point element type (",
        get_input_element_type(INPUT_ZERO_POINT),
        ") must match input element type (",
        get_input_element_type(0),
        ")");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(1).compatible(get_input_element_type(FILTER_ZERO_POINT)),
        "Filter Zero point element type (",
        get_input_element_type(FILTER_ZERO_POINT),
        ") must match filter element type (",
        get_input_element_type(1),
        ")");

    // TODO Remove these checks once we support channelwise and vector of scales
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(2).compatible(PartialShape{}) &&
                              get_input_partial_shape(3).compatible(PartialShape{}),
                          "Input scale and input zero point shape must be same and 1");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(4).compatible(PartialShape{}) &&
                              get_input_partial_shape(5).compatible(PartialShape{}),
                          "Filter scale and filter zero point shape must be same and 1");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(6).compatible(PartialShape{}) &&
                              get_input_partial_shape(7).compatible(PartialShape{}),
                          "Output scale and output zero point shape must be same and 1");

    // AxisSet should be empty till we support channel wise quantization
    NODE_VALIDATION_CHECK(this,
                          m_input_axes == AxisSet{} && m_filter_axes == AxisSet{} &&
                              m_output_axes == AxisSet{},
                          "Input, filter and output AxisSet should be empty");

    const PartialShape& input_shape = get_input_partial_shape(0);
    const PartialShape& filters_shape = get_input_partial_shape(1);

    PartialShape result_shape;

    result_shape = infer_convolution_forward(this,
                                             input_shape,
                                             m_data_dilation_strides,
                                             m_padding_below,
                                             m_padding_above,
                                             filters_shape,
                                             m_window_movement_strides,
                                             m_window_dilation_strides);

    NODE_VALIDATION_CHECK(
        this,
        get_output_element_type(0).compatible(get_input_element_type(OUTPUT_ZERO_POINT)),
        "Output Zero point element type (",
        get_input_element_type(OUTPUT_ZERO_POINT),
        ") must match output element type (",
        get_output_element_type(0),
        ")");

    set_output_type(0, m_output_type, result_shape);
}

shared_ptr<Node> op::QuantizedConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return shared_ptr<Node>(new QuantizedConvolution(new_args.at(0),
                                                     new_args.at(1),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides(),
                                                     new_args.at(2),
                                                     new_args.at(3),
                                                     new_args.at(4),
                                                     new_args.at(5),
                                                     new_args.at(6),
                                                     new_args.at(7),
                                                     m_output_type,
                                                     m_input_axes,
                                                     m_filter_axes,
                                                     m_output_axes));
}
