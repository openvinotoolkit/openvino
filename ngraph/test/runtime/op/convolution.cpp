//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "convolution.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

// *** Convolution OP SET 0 ***
constexpr NodeTypeInfo op::v0::Convolution::type_info;

op::v0::Convolution::Convolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& window_movement_strides,
                                 const Strides& window_dilation_strides,
                                 const CoordinateDiff& padding_below,
                                 const CoordinateDiff& padding_above,
                                 const Strides& data_dilation_strides,
                                 const PadType& pad_type)
    : Op({data_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_pad_type(pad_type)
{
    constructor_validate_and_infer_types();
}

bool op::v0::Convolution::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("window_movement_strides", m_window_movement_strides);
    visitor.on_attribute("window_dilation_strides", m_window_dilation_strides);
    visitor.on_attribute("data_dilation_strides", m_data_dilation_strides);
    visitor.on_attribute("padding_below", m_padding_below);
    visitor.on_attribute("padding_above", m_padding_above);
    visitor.on_attribute("pad_type", m_pad_type);
    return true;
}

void op::v0::Convolution::validate_and_infer_types()
{
    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& filters_shape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    if (m_data_dilation_strides.size() == 0)
    {
        m_data_dilation_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_window_movement_strides.size() == 0)
    {
        m_window_movement_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_window_dilation_strides.size() == 0)
    {
        m_window_dilation_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_padding_below.size() == 0)
    {
        m_padding_below = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_padding_above.size() == 0)
    {
        m_padding_above = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_pad_type == PadType::SAME_UPPER || m_pad_type == PadType::SAME_LOWER)
    {
        if (data_batch_shape.is_static() && filters_shape.is_static())
        {
            // TODO: data dilation
            m_padding_below.clear();
            m_padding_above.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            infer_auto_padding(data_batch_shape.to_shape(),
                               filter_shape,
                               m_window_movement_strides,
                               m_window_dilation_strides,
                               m_pad_type,
                               m_padding_above,
                               m_padding_below);
        }
    }

    element::Type result_et;
    PartialShape result_shape;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_batch_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        filters_et,
        ").");

    result_shape = infer_convolution_forward(this,
                                             data_batch_shape,
                                             m_data_dilation_strides,
                                             m_padding_below,
                                             m_padding_above,
                                             filters_shape,
                                             m_window_movement_strides,
                                             m_window_dilation_strides);

    set_output_type(0, result_et, result_shape);
}

op::v0::Convolution::Convolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& window_movement_strides,
                                 const Strides& window_dilation_strides,
                                 const CoordinateDiff& padding_below,
                                 const CoordinateDiff& padding_above)
    : Convolution(data_batch,
                  filters,
                  window_movement_strides,
                  window_dilation_strides,
                  padding_below,
                  padding_above,
                  Strides())
{
}

op::v0::Convolution::Convolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& window_movement_strides,
                                 const Strides& window_dilation_strides)
    : Convolution(data_batch,
                  filters,
                  window_movement_strides,
                  window_dilation_strides,
                  CoordinateDiff(),
                  CoordinateDiff())
{
}

op::v0::Convolution::Convolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& window_movement_strides)
    : Convolution(data_batch,
                  filters,
                  window_movement_strides,
                  Strides(),
                  CoordinateDiff(),
                  CoordinateDiff())
{
}

op::v0::Convolution::Convolution(const Output<Node>& data_batch, const Output<Node>& filters)
    : Convolution(data_batch, filters, Strides(), Strides(), CoordinateDiff(), CoordinateDiff())
{
}

shared_ptr<Node> op::v0::Convolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::Convolution>(new_args.at(0),
                                        new_args.at(1),
                                        m_window_movement_strides,
                                        m_window_dilation_strides,
                                        m_padding_below,
                                        m_padding_above,
                                        m_data_dilation_strides,
                                        m_pad_type);
}

constexpr NodeTypeInfo op::v0::ConvolutionBackpropData::type_info;
shared_ptr<Node> op::v0::Convolution::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

op::v0::ConvolutionBackpropData::ConvolutionBackpropData(
    const Shape& data_batch_shape,
    const Output<Node>& filters,
    const Output<Node>& output_delta,
    const Strides& window_movement_strides_forward,
    const Strides& window_dilation_strides_forward,
    const CoordinateDiff& padding_below_forward,
    const CoordinateDiff& padding_above_forward,
    const Strides& data_dilation_strides_forward)
    : Op({filters, output_delta})
    , m_data_batch_shape(data_batch_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    constructor_validate_and_infer_types();
}

bool op::v0::ConvolutionBackpropData::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("data_batch_shape", m_data_batch_shape);
    visitor.on_attribute("window_movement_strides_forward", m_window_movement_strides_forward);
    visitor.on_attribute("window_dilation_strides_forward", m_window_dilation_strides_forward);
    visitor.on_attribute("padding_below_forward", m_padding_below_forward);
    visitor.on_attribute("padding_above_forward", m_padding_above_forward);
    visitor.on_attribute("data_dilation_strides_forward", m_data_dilation_strides_forward);
    return true;
}

void op::v0::ConvolutionBackpropData::validate_and_infer_types()
{
    // Backprop to data is itself convolution, with inputs/outputs/attributes transmogrified as
    // follows.
    //
    //                          Forward   Backward
    // "N" axis for data batch  0         0
    // "C" axis for data batch  1         1
    // "Co" axis for filters    0         0
    // "Ci" axis for filters    1         1
    // "N" axis for output      0         0
    // "C" axis for output      1         1
    // Data batch               x         delta
    // Data batch shape         S_x       S_o
    // Filters                  f         reverse(f) [on spatial axes]
    // Filters shape            S_f       S_f
    // Window movement strides  q_x       p_x
    // Window dilation strides  p_f       p_f
    // Padding below            a_x       (S_f - 1)p_f - a_x
    // Padding above            b_x       (S_f - 1)p_f +
    //                                      + ((a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f)
    //                                         % q_x)
    //                                      - b_x
    // Data dilation strides    p_x       q_x
    // Output shape             S_o       S_x
    //
    // To _validate_, we simply need to check/infer the output shape of the forward convolution,
    // then check to make sure that the incoming delta has the same shape as the forward output.
    const PartialShape& filters_shape = get_input_partial_shape(0);
    element::Type filters_et = get_input_element_type(0);
    const PartialShape& delta_shape = get_input_partial_shape(1);
    element::Type delta_et = get_input_element_type(1);

    element::Type forward_result_et;
    PartialShape forward_result_shape;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(forward_result_et, delta_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        delta_et,
        ", filters element type: ",
        filters_et,
        ").");

    forward_result_shape = infer_convolution_forward(this,
                                                     m_data_batch_shape,
                                                     m_data_dilation_strides_forward,
                                                     m_padding_below_forward,
                                                     m_padding_above_forward,
                                                     filters_shape,
                                                     m_window_movement_strides_forward,
                                                     m_window_dilation_strides_forward);

    NODE_VALIDATION_CHECK(this,
                          forward_result_shape.compatible(delta_shape),
                          "Inferred forward output shape (",
                          forward_result_shape,
                          ") does not match shape of ",
                          "delta (",
                          delta_shape,
                          ").");

    set_output_type(0, forward_result_et, m_data_batch_shape);
}

shared_ptr<Node>
    op::v0::ConvolutionBackpropData::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::ConvolutionBackpropData>(m_data_batch_shape,
                                                    new_args.at(0),
                                                    new_args.at(1),
                                                    m_window_movement_strides_forward,
                                                    m_window_dilation_strides_forward,
                                                    m_padding_below_forward,
                                                    m_padding_above_forward,
                                                    m_data_dilation_strides_forward);
}

CoordinateDiff op::v0::ConvolutionBackpropData::compute_backward_delta_out_pad_below() const
{
    auto& in_shape = get_data_batch_shape();
    auto& filter_dilation = get_window_dilation_strides_forward();
    auto& filter_shape = get_input_shape(0);
    auto& in_pad_below = get_padding_below_forward();
    size_t spatial_dim_count = static_cast<size_t>(in_shape.size()) - 2;

    CoordinateDiff backward_delta_out_pad_below;
    backward_delta_out_pad_below.resize(spatial_dim_count);

    for (size_t i = 0; i < spatial_dim_count; i++)
    {
        backward_delta_out_pad_below[i] =
            (static_cast<ptrdiff_t>(filter_shape[i + 2]) - 1) * filter_dilation[i] -
            in_pad_below[i];
    }
    return backward_delta_out_pad_below;
}

CoordinateDiff op::v0::ConvolutionBackpropData::compute_backward_delta_out_pad_above() const
{
    auto& in_shape = get_data_batch_shape();
    auto& filter_dilation = get_window_dilation_strides_forward();
    auto& filter_shape = get_input_shape(0);
    auto& in_pad_below = get_padding_below_forward();
    auto& in_pad_above = get_padding_above_forward();
    auto& in_dilation = get_data_dilation_strides_forward();
    auto& stride = get_window_movement_strides_forward();
    size_t spatial_dim_count = static_cast<size_t>(in_shape.size()) - 2;

    CoordinateDiff backward_delta_out_pad_above;
    backward_delta_out_pad_above.resize(spatial_dim_count);

    for (size_t i = 0; i < spatial_dim_count; i++)
    {
        backward_delta_out_pad_above[i] =
            (static_cast<ptrdiff_t>(filter_shape[i + 2]) - 1) * filter_dilation[i] +
            ((in_pad_below[i] + ((in_shape[i + 2]) - 1) * in_dilation[i] + in_pad_above[i] -
              (static_cast<ptrdiff_t>(filter_shape[i + 2]) - 1) * filter_dilation[i]) %
             stride[i]) -
            in_pad_above[i];
    }
    return backward_delta_out_pad_above;
}
