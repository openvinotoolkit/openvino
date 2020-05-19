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

#include "quantized_dot.hpp"
#include <numeric>
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::QuantizedDot::type_info;

op::QuantizedDot::QuantizedDot(const Output<Node>& input0,
                               const Output<Node>& input1,
                               size_t reduction_axes_count,
                               const Output<Node>& input0_scale,
                               const Output<Node>& input0_zero_point,
                               const Output<Node>& input1_scale,
                               const Output<Node>& input1_zero_point,
                               const Output<Node>& output_scale,
                               const Output<Node>& output_zero_point,
                               const element::Type& output_type,
                               const AxisSet& input0_axes,
                               const AxisSet& input1_axes,
                               const AxisSet& output_axes)
    : Op({input0,
          input1,
          input0_scale,
          input0_zero_point,
          input1_scale,
          input1_zero_point,
          output_scale,
          output_zero_point})
    , m_reduction_axes_count(reduction_axes_count)
    , m_output_type(output_type)
    , m_input0_axes(input0_axes)
    , m_input1_axes(input1_axes)
    , m_output_axes(output_axes)
{
    constructor_validate_and_infer_types();
}

void op::QuantizedDot::validate_and_infer_types()
{
    enum
    {
        INPUT0,
        INPUT1,
        INPUT0_SCALE,
        INPUT0_ZERO_POINT,
        INPUT1_SCALE,
        INPUT1_ZERO_POINT,
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
                          get_input_element_type(INPUT0).is_quantized(),
                          "Input0 element type (",
                          get_input_element_type(INPUT0),
                          ") must be a quantized type");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INPUT1).is_quantized(),
                          "Input1 element type (",
                          get_input_element_type(INPUT1),
                          ") must be a quantized type");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INPUT0_SCALE).is_real() ||
                              get_input_element_type(INPUT0_SCALE).is_dynamic() ||
                              get_input_element_type(INPUT1_SCALE).is_real() ||
                              get_input_element_type(INPUT1_SCALE).is_dynamic() ||
                              get_input_element_type(OUTPUT_SCALE).is_real() ||
                              get_input_element_type(OUTPUT_SCALE).is_dynamic(),
                          "Scale must be a floating point number");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INPUT0).compatible(get_input_element_type(INPUT0_ZERO_POINT)),
        "Input0 Zero point element type (",
        get_input_element_type(INPUT0_ZERO_POINT),
        ") must match input0 element type (",
        get_input_element_type(INPUT0),
        ")");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INPUT1).compatible(get_input_element_type(INPUT1_ZERO_POINT)),
        "Input1 Zero point element type (",
        get_input_element_type(INPUT1_ZERO_POINT),
        ") must match input1 element type (",
        get_input_element_type(INPUT1),
        ")");

    // TODO Remove these checks once we support channelwise and vector of scales
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(2).compatible(PartialShape{}) &&
                              get_input_partial_shape(3).compatible(PartialShape{}),
                          "Input0 scale and input0 zero point shape must be same and 1");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(4).compatible(PartialShape{}) &&
                              get_input_partial_shape(5).compatible(PartialShape{}),
                          "Input1 scale and input1 zero point shape must be same and 1");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(6).compatible(PartialShape{}) &&
                              get_input_partial_shape(7).compatible(PartialShape{}),
                          "Output scale and output zero point shape must be same and 1");

    // AxisSet should be empty till we support channel wise quantization
    NODE_VALIDATION_CHECK(this,
                          m_input0_axes == AxisSet{} && m_input1_axes == AxisSet{} &&
                              m_output_axes == AxisSet{},
                          "Input0, input1 and output AxisSet should be empty");

    const PartialShape& arg0_shape = get_input_partial_shape(0);
    const PartialShape& arg1_shape = get_input_partial_shape(1);

    PartialShape result_shape;

    if (arg0_shape.rank().is_static() && arg1_shape.rank().is_static())
    {
        for (size_t i = 0; i < m_reduction_axes_count; i++)
        {
            size_t axis_index_arg0 = arg0_shape.rank().get_length() - m_reduction_axes_count + i;
            size_t axis_index_arg1 = i;

            NODE_VALIDATION_CHECK(
                this,
                arg0_shape[axis_index_arg0].compatible(arg1_shape[axis_index_arg1]),
                "Paired axes (axis ",
                axis_index_arg0,
                " from arg0, axis ",
                axis_index_arg1,
                " from arg1) do not have same length (arg0 shape: ",
                arg0_shape,
                ", arg1 shape: ",
                arg1_shape,
                ", reduction axes count: ",
                m_reduction_axes_count,
                ").");
        }

        std::vector<Dimension> result_dims(arg0_shape.rank().get_length() +
                                           arg1_shape.rank().get_length() -
                                           2 * m_reduction_axes_count);

        size_t i = 0;

        for (size_t j = 0; j < arg0_shape.rank().get_length() - m_reduction_axes_count; j++)
        {
            result_dims[i++] = arg0_shape[j];
        }
        for (size_t j = m_reduction_axes_count; j < arg1_shape.rank().get_length(); j++)
        {
            result_dims[i++] = arg1_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

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

shared_ptr<Node> op::QuantizedDot::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return shared_ptr<Node>(new QuantizedDot(new_args.at(0),
                                             new_args.at(1),
                                             m_reduction_axes_count,
                                             new_args.at(2),
                                             new_args.at(3),
                                             new_args.at(4),
                                             new_args.at(5),
                                             new_args.at(6),
                                             new_args.at(7),
                                             m_output_type,
                                             m_input0_axes,
                                             m_input1_axes,
                                             m_output_axes));
}

void op::QuantizedDot::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                         const OutputVector& /* deltas */)
{
    throw ngraph_error("Forward-propagation-only operation");
}
