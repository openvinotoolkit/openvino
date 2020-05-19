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

#include "conv_fused.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ConvolutionBias::type_info;
constexpr NodeTypeInfo op::ConvolutionBiasBackpropFiltersBias::type_info;
constexpr NodeTypeInfo op::ConvolutionBiasAdd::type_info;

static void validate_convbias_shapes(const Node* node,
                                     element::Type et_filters,
                                     element::Type et_bias,
                                     const PartialShape& filters_shape,
                                     const PartialShape& bias_shape)
{
    element::Type et_result;

    NODE_VALIDATION_CHECK(node,
                          element::Type::merge(et_result, et_bias, et_filters),
                          "Element types for bias and filters do not match (bias element type: ",
                          et_bias,
                          ", filters element type: ",
                          et_filters,
                          ").");

    NODE_VALIDATION_CHECK(node,
                          bias_shape.rank().is_dynamic() || bias_shape.rank().get_length() == 1,
                          "Bias must have a rank of 1 (bias_shape: ",
                          bias_shape,
                          ").");

    if (bias_shape.rank().is_static() && filters_shape.rank().is_static())
    {
        Dimension filter_count;
        NODE_VALIDATION_CHECK(node,
                              Dimension::merge(filter_count, bias_shape[0], filters_shape[0]),
                              "Bias channel count (",
                              bias_shape[0],
                              ") does not match filter output channel count (",
                              filters_shape[0],
                              ").");
    }
}

op::ConvolutionBias::ConvolutionBias(const Output<Node>& data_batch,
                                     const Output<Node>& filters,
                                     const Output<Node>& bias,
                                     const Strides& window_movement_strides,
                                     const Strides& window_dilation_strides,
                                     const CoordinateDiff& padding_below,
                                     const CoordinateDiff& padding_above,
                                     const Strides& data_dilation_strides,
                                     const bool with_relu)
    : FusedOp({data_batch, filters, bias})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();
}

op::ConvolutionBias::ConvolutionBias(const shared_ptr<op::Convolution>& conv,
                                     const Output<Node>& bias,
                                     const bool with_relu)
    : ConvolutionBias(conv->input_value(0),
                      conv->input_value(1),
                      bias,
                      conv->get_window_movement_strides(),
                      conv->get_window_dilation_strides(),
                      conv->get_padding_below(),
                      conv->get_padding_above(),
                      conv->get_data_dilation_strides(),
                      with_relu)
{
}

op::ConvolutionBias::ConvolutionBias(const Output<Node>& data_batch,
                                     const Output<Node>& filters,
                                     const Output<Node>& bias)
    : ConvolutionBias(data_batch,
                      filters,
                      bias,
                      Strides(),
                      Strides(),
                      CoordinateDiff(),
                      CoordinateDiff(),
                      Strides())
{
}

/// Overrides the default shape inference utility provided by FusedOp
/// based on FusedOp decomposition.
///
/// This implementation handles partial shapes and adjust conv attributes
/// to support simplified ConvolutionBias op construction
void op::ConvolutionBias::validate_and_infer_types()
{
    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& filters_shape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);
    const PartialShape& bias_shape = get_input_partial_shape(2);
    element::Type bias_et = get_input_element_type(2);

    validate_convbias_shapes(this, filters_et, bias_et, filters_shape, bias_shape);

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

shared_ptr<Node> op::ConvolutionBias::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<ConvolutionBias>(new_args.at(0),
                                        new_args.at(1),
                                        new_args.at(2),
                                        get_window_movement_strides(),
                                        get_window_dilation_strides(),
                                        get_padding_below(),
                                        get_padding_above(),
                                        get_data_dilation_strides(),
                                        m_with_relu);
}

NodeVector op::ConvolutionBias::decompose_op() const
{
    auto conv = make_shared<op::Convolution>(input_value(0),
                                             input_value(1),
                                             m_window_movement_strides,
                                             m_window_dilation_strides,
                                             m_padding_below,
                                             m_padding_above,
                                             m_data_dilation_strides);
    AxisSet bcast_axes;
    bcast_axes.insert(0);
    for (size_t i = 2; i < conv->get_shape().size(); i++)
    {
        bcast_axes.insert(i);
    }

    auto conv_bias = make_shared<op::Add>(
        conv, make_shared<op::Broadcast>(input_value(2), conv->get_shape(), bcast_axes));
    if (m_with_relu)
    {
        return {make_shared<op::Relu>(conv_bias)};
    }
    else
    {
        return {conv_bias};
    }
}

void op::ConvolutionBias::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const OutputVector& deltas)
{
    auto delta = deltas.at(0);
    if (m_with_relu)
    {
        delta = make_shared<op::ReluBackprop>(shared_from_this(), delta);
    }

    auto data = input_value(0);
    const auto data_shape = data.get_shape();

    auto filter = input_value(1);
    const auto filter_shape = filter.get_shape();

    auto bias = input_value(2);
    const auto bias_shape = bias.get_shape();

    // using regular convolution backprop for data
    adjoints.add_delta(data,
                       make_shared<op::ConvolutionBackpropData>(data_shape,
                                                                filter,
                                                                delta,
                                                                m_window_movement_strides,
                                                                m_window_dilation_strides,
                                                                m_padding_below,
                                                                m_padding_above,
                                                                m_data_dilation_strides));

    auto filter_bias_backprop =
        make_shared<op::ConvolutionBiasBackpropFiltersBias>(data,
                                                            filter_shape,
                                                            bias_shape,
                                                            delta,
                                                            m_window_movement_strides,
                                                            m_window_dilation_strides,
                                                            m_padding_below,
                                                            m_padding_above,
                                                            m_data_dilation_strides);
    auto filter_delta = Output<Node>(filter_bias_backprop, 0);
    auto bias_delta = Output<Node>(filter_bias_backprop, 1);

    adjoints.add_delta(filter, filter_delta);
    adjoints.add_delta(bias, bias_delta);
}

op::ConvolutionBiasBackpropFiltersBias::ConvolutionBiasBackpropFiltersBias(
    const Output<Node>& data_batch,
    const Shape& filters_shape,
    const Shape& bias_shape,
    const Output<Node>& output_delta,
    const Strides& window_movement_strides_forward,
    const Strides& window_dilation_strides_forward,
    const CoordinateDiff& padding_below_forward,
    const CoordinateDiff& padding_above_forward,
    const Strides& data_dilation_strides_forward)
    : FusedOp({data_batch, output_delta})
    , m_filters_shape(filters_shape)
    , m_bias_shape(bias_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    auto& data_batch_shape = get_input_shape(0);

    //                              Forward               Backward
    // Window movement strides      q                     p_f
    // Window dilation strides      p_f                   q
    // Padding below                a_x                   a_x
    // Padding above                b_x                   b_x -
    //                                                      (a_x + (S_x - 1)p_x + b_x -
    //                                                        (S_f - 1)p_f)
    //                                                       % q
    // Data dilation strides        p_x                   p_x

    for (size_t i = 0; i < filters_shape.size() - 2; i++)
    {
        m_window_movement_strides_backward.push_back(window_dilation_strides_forward[i]);
        m_window_dilation_strides_backward.push_back(window_movement_strides_forward[i]);
        m_padding_below_backward.push_back(padding_below_forward[i]);
        m_padding_above_backward.push_back(
            padding_above_forward[i] -
            (padding_below_forward[i] +
             (data_batch_shape[i + 2] - 1) * data_dilation_strides_forward[i] +
             padding_above_forward[i] -
             (filters_shape[i + 2] - 1) * window_dilation_strides_forward[i]) %
                window_movement_strides_forward[i]);
        m_data_dilation_strides_backward.push_back(data_dilation_strides_forward[i]);
    }

    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::ConvolutionBiasBackpropFiltersBias::clone_with_new_inputs(
    const OutputVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<ConvolutionBiasBackpropFiltersBias>(new_args.at(0),
                                                           m_filters_shape,
                                                           m_bias_shape,
                                                           new_args.at(1),
                                                           m_window_movement_strides_forward,
                                                           m_window_dilation_strides_forward,
                                                           m_padding_below_forward,
                                                           m_padding_above_forward,
                                                           m_data_dilation_strides_forward);
}

NodeVector op::ConvolutionBiasBackpropFiltersBias::decompose_op() const
{
    auto conv_bprop = make_shared<op::ConvolutionBackpropFilters>(input_value(0),
                                                                  m_filters_shape,
                                                                  input_value(1),
                                                                  m_window_movement_strides_forward,
                                                                  m_window_dilation_strides_forward,
                                                                  m_padding_below_forward,
                                                                  m_padding_above_forward,
                                                                  m_data_dilation_strides_forward);

    AxisSet reduce_axes;
    reduce_axes.insert(0);
    for (size_t i = 2; i < conv_bprop->get_shape().size(); i++)
    {
        reduce_axes.insert(i);
    }

    auto bias_bprop = make_shared<op::Sum>(input_value(1), reduce_axes);

    return {conv_bprop, bias_bprop};
}

op::ConvolutionBiasAdd::ConvolutionBiasAdd(const Output<Node>& data_batch,
                                           const Output<Node>& filters,
                                           const Output<Node>& bias,
                                           const Output<Node>& add_input,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           bool with_relu)
    : FusedOp({data_batch, filters, bias, add_input})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();
}

op::ConvolutionBiasAdd::ConvolutionBiasAdd(const std::shared_ptr<op::ConvolutionBias>& conv,
                                           const Output<Node>& add_input,
                                           bool with_relu)
    : ConvolutionBiasAdd(conv->input_value(0),
                         conv->input_value(1),
                         conv->input_value(2),
                         add_input,
                         conv->get_window_movement_strides(),
                         conv->get_window_dilation_strides(),
                         conv->get_padding_below(),
                         conv->get_padding_above(),
                         conv->get_data_dilation_strides(),
                         with_relu)
{
}

/// Overrides the default shape inference utility provided by FusedOp
/// based on FusedOp decomposition.
///
/// This implementation handles partial shapes
void op::ConvolutionBiasAdd::validate_and_infer_types()
{
    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& filters_shape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);
    const PartialShape& bias_shape = get_input_partial_shape(2);
    element::Type bias_et = get_input_element_type(2);

    validate_convbias_shapes(this, filters_et, bias_et, filters_shape, bias_shape);

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
    // TODO: Check result_shape is compatible with add_input
    set_output_type(0, result_et, result_shape);
}

std::shared_ptr<Node>
    op::ConvolutionBiasAdd::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 4)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<ConvolutionBiasAdd>(new_args.at(0),
                                           new_args.at(1),
                                           new_args.at(2),
                                           new_args.at(3),
                                           get_window_movement_strides(),
                                           get_window_dilation_strides(),
                                           get_padding_below(),
                                           get_padding_above(),
                                           get_data_dilation_strides(),
                                           m_with_relu);
}

NodeVector op::ConvolutionBiasAdd::decompose_op() const
{
    auto conv = make_shared<op::Convolution>(input_value(0),
                                             input_value(1),
                                             m_window_movement_strides,
                                             m_window_dilation_strides,
                                             m_padding_below,
                                             m_padding_above,
                                             m_data_dilation_strides);
    AxisSet bcast_axes;
    bcast_axes.insert(0);
    for (size_t i = 2; i < conv->get_shape().size(); i++)
    {
        bcast_axes.insert(i);
    }

    auto conv_bias = make_shared<op::Add>(
        conv, make_shared<op::Broadcast>(input_value(2), conv->get_shape(), bcast_axes));
    if (m_with_relu)
    {
        return {make_shared<op::Relu>(conv_bias + input_value(3))};
    }
    else
    {
        return {conv_bias + input_value(3)};
    }
}
