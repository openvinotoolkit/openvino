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

#include <numeric>

#include "convolution.hpp"
#include "group_conv.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

//------------------------------------------------------------------------------
//                        v0::GroupConvolution
//------------------------------------------------------------------------------

constexpr NodeTypeInfo op::v0::GroupConvolution::type_info;

op::v0::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
                                           const Output<Node>& filters,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           const size_t groups,
                                           const PadType& pad_type)
    : FusedOp({data_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_groups(groups)
    , m_pad_type(pad_type)
    , m_groups_in_filters(false)
{
    constructor_validate_and_infer_types();
}

op::v0::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
                                           const Output<Node>& filters,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           const PadType& pad_type)
    : FusedOp({data_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_groups(0)
    , m_pad_type(pad_type)
    , m_groups_in_filters(true)
{
    constructor_validate_and_infer_types();
}

void op::v0::GroupConvolution::pre_validate_and_infer_types()
{
    auto data_shape = get_input_partial_shape(0);
    auto filters_shape = get_input_partial_shape(1);

    if (data_shape.is_static() && filters_shape.is_static())
    {
        // Update groups
        if (m_groups_in_filters)
        {
            m_groups = get_input_partial_shape(1)[0].get_length();
        }

        // Data channels
        NODE_VALIDATION_CHECK(this,
                              data_shape.to_shape()[1] % get_groups() == 0,
                              "Data channels not a multiple of group size");
        // Output channels
        NODE_VALIDATION_CHECK(this,
                              filters_shape.to_shape()[0] % get_groups() == 0,
                              "# Filters not a multiple of group size");

        // Input Filters
        NODE_VALIDATION_CHECK(this,
                              (filters_shape.to_shape()[m_groups_in_filters ? 2 : 1] *
                               get_groups()) == data_shape.to_shape()[1],
                              "Incorrect number of channels per filter");
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

void op::v0::GroupConvolution::post_validate_and_infer_types()
{
    auto data_shape = get_input_partial_shape(0);
    auto filters_shape = get_input_partial_shape(1);
    if (data_shape.is_static() && filters_shape.is_static())
    {
        if (m_pad_type == PadType::SAME_UPPER || m_pad_type == PadType::SAME_LOWER)
        {
            m_padding_below.clear();
            m_padding_above.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            infer_auto_padding(data_shape.to_shape(),
                               filter_shape,
                               m_window_movement_strides,
                               m_window_dilation_strides,
                               m_pad_type,
                               m_padding_above,
                               m_padding_below);
        }
    }
}

Shape op::v0::GroupConvolution::get_weights_dimensions() const
{
    auto data_shape = get_input_shape(0);
    auto weights_shape = get_input_shape(1);
    // check if weights already includes groups
    if (m_groups_in_filters)
    {
        return weights_shape;
    }
    // reshape weights into 5d tensors that includes groups
    const size_t OC = 0;
    const size_t OC_IN_OUTPUT = 1;
    const size_t IC = 1;
    Shape weights_shape_groups{weights_shape};
    // adjust output and channel given a number of groups

    weights_shape_groups.at(OC) = get_shape().at(OC_IN_OUTPUT) / get_groups();
    weights_shape_groups.at(IC) = data_shape.at(IC) / get_groups();
    // push_front the number of groups
    weights_shape_groups.insert(weights_shape_groups.begin(), get_groups());
    return weights_shape_groups;
}

shared_ptr<Node> op::v0::GroupConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);

    if (m_groups_in_filters)
    {
        return make_shared<op::v0::GroupConvolution>(new_args.at(0),
                                                     new_args.at(1),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides(),
                                                     get_pad_type());
    }
    else
    {
        return make_shared<op::v0::GroupConvolution>(new_args.at(0),
                                                     new_args.at(1),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides(),
                                                     get_groups(),
                                                     get_pad_type());
    }
}

OutputVector op::v0::GroupConvolution::decompose_op() const
{
    auto data = input_value(0);
    auto filters = input_value(1);
    auto filters_shape = get_input_shape(1);
    // Split one convolution op to N ops where N is the number of groups
    // and concat results after computation.
    NodeVector convolution_nodes;

    // slice data
    auto sliced_data = builder::split(data, get_groups(), 1);
    // slice filters
    auto sliced_filters = builder::split(filters, get_groups(), 0);
    auto shape = Shape(std::next(std::begin(filters_shape), 1), std::end(filters_shape));
    for (std::size_t group{0}; group < get_groups(); ++group)
    {
        auto sliced_filter = sliced_filters[group];
        if (m_groups_in_filters)
        {
            // Remove group dimension after slicing
            sliced_filter = builder::opset1::reshape(sliced_filters[group], shape);
        }
        convolution_nodes.push_back(
            std::make_shared<ngraph::op::v0::Convolution>(sliced_data[group],
                                                          sliced_filter,
                                                          m_window_movement_strides,
                                                          m_window_dilation_strides,
                                                          m_padding_below,
                                                          m_padding_above,
                                                          m_data_dilation_strides,
                                                          m_pad_type));
    }
    std::size_t concatenation_axis = 1;
    return {std::make_shared<ngraph::op::Concat>(convolution_nodes, concatenation_axis)};
}

//------------------------------------------------------------------------------
//                        v0::GroupConvolutionBackpropData
//------------------------------------------------------------------------------

constexpr NodeTypeInfo op::v0::GroupConvolutionBackpropData::type_info;

op::v0::GroupConvolutionBackpropData::GroupConvolutionBackpropData(
    const Output<Node>& data_batch,
    const Output<Node>& filters,
    const Output<Node>& output_delta,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above,
    const size_t groups)
    : FusedOp({data_batch, filters, output_delta})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_groups(groups)
{
    constructor_validate_and_infer_types();
}

void op::v0::GroupConvolutionBackpropData::pre_validate_and_infer_types()
{
    element::Type data_element_type = get_input_element_type(2);
    element::Type filters_elem_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          data_element_type.is_dynamic() || data_element_type.is_real(),
                          "Output delta element type must be f16, bf16, f32, f64 or dynamic (got ",
                          data_element_type,
                          ").");
    NODE_VALIDATION_CHECK(this,
                          filters_elem_type.is_dynamic() || filters_elem_type.is_real(),
                          "Filters element type must be f16, bf16, f32, f64 or dynamic (got ",
                          filters_elem_type,
                          ").");

    PartialShape data_pshape = get_input_partial_shape(0);
    PartialShape filters_pshape = get_input_partial_shape(1);
    PartialShape delta_pshape = get_input_partial_shape(2);

    if (data_pshape.is_dynamic() || filters_pshape.is_dynamic() || delta_pshape.is_dynamic())
    {
        set_output_type(0, data_element_type, PartialShape::dynamic());
    }
}

shared_ptr<Node>
    op::v0::GroupConvolutionBackpropData::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<op::v0::GroupConvolutionBackpropData>(new_args.at(0),
                                                             new_args.at(1),
                                                             new_args.at(2),
                                                             get_window_movement_strides(),
                                                             get_window_dilation_strides(),
                                                             get_padding_below(),
                                                             get_padding_above(),
                                                             get_groups());
}

OutputVector op::v0::GroupConvolutionBackpropData::decompose_op() const
{
    auto filters = input_value(1);
    auto output_delta = input_value(2);
    auto data_shape = get_input_shape(0);

    NodeVector sliced_inputs;

    auto groups = get_groups();
    // slice data shape
    data_shape[1] /= groups;
    // slice delta
    auto sliced_delta = builder::split(output_delta, groups, 1);
    // slice filters
    auto sliced_filters = builder::split(filters, groups, 0);

    auto num_spatials = get_window_movement_strides().size();

    for (size_t i = 0; i < groups; ++i)
    {
        auto sliced_conv = std::make_shared<op::v0::ConvolutionBackpropData>(
            data_shape,
            sliced_filters[i],
            sliced_delta[i],
            get_window_movement_strides(),
            get_window_dilation_strides(),
            get_padding_below(),
            get_padding_above(),
            Strides(num_spatials, 1)); // default data dilation strides

        sliced_inputs.push_back(sliced_conv);
    }

    size_t concatenation_axis = 1;
    return {std::make_shared<ngraph::op::Concat>(sliced_inputs, concatenation_axis)};
}
