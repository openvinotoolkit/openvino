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

#include "ngraph/op/convolution.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

// *** Convolution OP SET 1 ***
NGRAPH_RTTI_DEFINITION(op::v1::Convolution, "Convolution", 1);

op::v1::Convolution::Convolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& strides,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const Strides& dilations,
                                 const PadType& auto_pad)
    : Op({data_batch, filters})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
{
    constructor_validate_and_infer_types();
}

bool op::v1::Convolution::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::v1::Convolution::validate_and_infer_types()
{
    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& filters_shape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    PartialShape result_shape = PartialShape::dynamic();
    if (data_batch_shape.rank().is_static())
    {
        result_shape =
            std::vector<Dimension>(data_batch_shape.rank().get_length(), Dimension::dynamic());

        if (data_batch_shape.rank().get_length() > 1)
        {
            result_shape[0] = data_batch_shape[0]; // batch size
        }
        if (filters_shape.rank().is_static() && filters_shape.rank().get_length() > 1)
        {
            result_shape[1] = filters_shape[0]; // filter channel size
        }
    }

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_batch_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        filters_et,
        ").");

    if (m_strides.size() == 0)
    {
        m_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_dilations.size() == 0)
    {
        m_dilations = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_pads_begin.size() == 0 || m_auto_pad == PadType::VALID)
    {
        m_pads_begin = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_pads_end.size() == 0 || m_auto_pad == PadType::VALID)
    {
        m_pads_end = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        bool auto_padding_applied = false;
        if (filters_shape.is_static())
        {
            m_pads_begin.clear();
            m_pads_end.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            auto_padding_applied = try_apply_auto_padding(data_batch_shape,
                                                          filter_shape,
                                                          m_strides,
                                                          m_dilations,
                                                          m_auto_pad,
                                                          m_pads_end,
                                                          m_pads_begin);
        }
        if (!auto_padding_applied)
        {
            set_output_type(0, result_et, result_shape);
            return;
        }
    }

    result_shape = infer_convolution_forward(this,
                                             data_batch_shape,
                                             Strides(m_strides.size(), 1), // dummy data dilations
                                             m_pads_begin,
                                             m_pads_end,
                                             filters_shape,
                                             m_strides,
                                             m_dilations);

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::v1::Convolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Convolution>(new_args.at(0),
                                        new_args.at(1),
                                        m_strides,
                                        m_pads_begin,
                                        m_pads_end,
                                        m_dilations,
                                        m_auto_pad);
}

constexpr NodeTypeInfo op::v1::ConvolutionBackpropData::type_info;
shared_ptr<Node> op::v1::Convolution::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

op::v1::ConvolutionBackpropData::ConvolutionBackpropData(const Output<Node>& data,
                                                         const Output<Node>& filters,
                                                         const Output<Node>& output_shape,
                                                         const Strides& strides,
                                                         const CoordinateDiff& pads_begin,
                                                         const CoordinateDiff& pads_end,
                                                         const Strides& dilations,
                                                         const PadType& auto_pad,
                                                         const CoordinateDiff& output_padding)
    : Op({data, filters, output_shape})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
    , m_output_padding(output_padding)
{
    constructor_validate_and_infer_types();
}

bool op::v1::ConvolutionBackpropData::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("output_padding", m_output_padding);
    return true;
}

op::v1::ConvolutionBackpropData::ConvolutionBackpropData(const Output<Node>& data,
                                                         const Output<Node>& filters,
                                                         const Strides& strides,
                                                         const CoordinateDiff& pads_begin,
                                                         const CoordinateDiff& pads_end,
                                                         const Strides& dilations,
                                                         const PadType& auto_pad,
                                                         const CoordinateDiff& output_padding)
    : Op({data, filters})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
    , m_output_padding(output_padding)
{
    constructor_validate_and_infer_types();
}

bool op::v1::ConvolutionBackpropData::is_dynamic() const
{
    bool is_dynamic = Node::is_dynamic();
    if (inputs().size() == 3 && !is_dynamic)
    {
        return !is_type<op::Constant>(input_value(2).get_node());
    }
    return is_dynamic;
}

const PartialShape op::v1::ConvolutionBackpropData::get_output_shape() const
{
    auto data_pshape = get_input_partial_shape(0);

    PartialShape shape;
    if (data_pshape.rank().is_static())
    {
        shape = PartialShape{vector<Dimension>(data_pshape.rank().get_length() - 2)};
    }
    else
    {
        shape = PartialShape{vector<Dimension>(m_strides.size())};
    }
    bool is_output_shape_present = inputs().size() == 3;
    if (is_output_shape_present)
    {
        if (auto const_op = as_type<op::Constant>(input_value(2).get_node()))
        {
            shape = const_op->get_shape_val();
        }
        else
        {
            shape = PartialShape::dynamic();
        }
    }
    return shape;
}

void op::v1::ConvolutionBackpropData::set_output_shape(const Shape& shape)
{
    this->input(2).replace_source_output(
        op::Constant::create(this->get_input_element_type(2), Shape{shape.size()}, shape)
            ->output(0));
}

void op::v1::ConvolutionBackpropData::infer_conv_backprop_output_spatial_shape(
    const vector<Dimension>& input_data_shape,
    const vector<Dimension>& filters_shape,
    const Strides& strides,
    const Strides& dilations,
    const CoordinateDiff& pads_begin,
    const CoordinateDiff& pads_end,
    const CoordinateDiff& output_padding,
    vector<Dimension>& output_spatial_shape)
{
    size_t num_spatial_dims = input_data_shape.size();
    NODE_VALIDATION_CHECK(
        this,
        filters_shape.size() == num_spatial_dims && strides.size() == num_spatial_dims &&
            dilations.size() == num_spatial_dims && pads_begin.size() == num_spatial_dims &&
            pads_end.size() == num_spatial_dims && output_padding.size() == num_spatial_dims);

    for (size_t i = 0; i < num_spatial_dims; ++i)
    {
        if (input_data_shape[i].is_static() && filters_shape[i].is_static())
        {
            int64_t val = strides[i] * (input_data_shape[i].get_length() - 1) +
                          dilations[i] * (filters_shape[i].get_length() - 1) + 1 - pads_begin[i] -
                          pads_end[i] + output_padding[i];
            output_spatial_shape.push_back(val);
        }
        else
        {
            output_spatial_shape.push_back(Dimension::dynamic());
        }
    }
}

void op::v1::ConvolutionBackpropData::validate_and_infer_types()
{
    auto data_pshape = get_input_partial_shape(0);
    element::Type delta_et = get_input_element_type(0);
    const PartialShape& filters_pshape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    bool is_output_shape_present = inputs().size() == 3;
    PartialShape output_pshape = get_output_shape();

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, delta_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        delta_et,
        ", filters element type: ",
        filters_et,
        ").");

    if (data_pshape.rank().is_static() && filters_pshape.rank().is_static())
    {
        if (m_pads_begin.size() == 0)
        {
            m_pads_begin = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (m_pads_end.size() == 0)
        {
            m_pads_end = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (m_output_padding.size() == 0)
        {
            m_output_padding = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (m_strides.size() == 0)
        {
            m_strides = conv_default_strides(this, data_pshape, filters_pshape);
        }
        if (m_dilations.size() == 0)
        {
            m_dilations = conv_default_strides(this, data_pshape, filters_pshape);
        }

        const auto num_spatial_dims = data_pshape.rank().get_length() - 2;

        NODE_VALIDATION_CHECK(this,
                              m_strides.size() == num_spatial_dims,
                              "Strides should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_dilations.size() == num_spatial_dims,
                              "Dilations should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_output_padding.size() == num_spatial_dims,
                              "Output padding should be defined for all and only "
                              "spatial features.");
    }

    PartialShape result_shape;
    if (is_output_shape_present)
    {
        if (output_pshape.is_static() && filters_pshape.is_static() && data_pshape.is_static())
        {
            Shape output_shape = output_pshape.to_shape();
            const Shape data_shape = data_pshape.to_shape();
            const Shape filters_shape = filters_pshape.to_shape();
            const size_t num_spatial_dims = data_shape.size() - 2;

            NODE_VALIDATION_CHECK(this,
                                  output_shape.size() == num_spatial_dims,
                                  "Output shape should be specified only and for "
                                  "all spatial dimensions.");

            // If auto_pad has one of following mode we infer paddings. Otherwise in
            // EXPLICIT auto_pad mode we use what is provided.
            if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
            {
                opset1::infer_conv_backprop_auto_padding(
                    Shape{std::next(data_shape.begin(), 2), std::end(data_shape)},
                    Shape{std::next(filters_shape.begin(), 2), std::end(filters_shape)},
                    output_shape,
                    m_strides,
                    m_dilations,
                    m_auto_pad,
                    m_output_padding,
                    m_pads_begin,
                    m_pads_end);
            }

            // C_OUTPUT
            output_shape.insert(output_shape.begin(), filters_shape.at(1));
            // N
            output_shape.insert(output_shape.begin(), data_shape.at(0));
            output_pshape = output_shape;
        }
        set_input_is_relevant_to_shape(2);
    }
    // Deduce output shape from input spatial shape, strides, dilations, output padding
    // and padding values.
    else
    {
        if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER ||
            m_auto_pad == PadType::VALID)
        {
            m_pads_begin.assign(m_pads_begin.size(), 0);
            m_pads_end.assign(m_pads_end.size(), 0);
        }

        if (data_pshape.rank().is_static() && filters_pshape.is_static())
        {
            vector<Dimension> data_shape{data_pshape}, filters_shape{filters_pshape}, output_shape;

            infer_conv_backprop_output_spatial_shape(
                vector<Dimension>{std::next(data_shape.begin(), 2), std::end(data_shape)},
                vector<Dimension>{std::next(filters_shape.begin(), 2), std::end(filters_shape)},
                m_strides,
                m_dilations,
                m_pads_begin,
                m_pads_end,
                m_output_padding,
                output_shape);

            // C_OUTPUT
            output_shape.insert(output_shape.begin(), filters_shape.at(1));
            // N
            output_shape.insert(output_shape.begin(), data_shape.at(0));
            output_pshape = PartialShape{output_shape};
        }
        else
        {
            output_pshape = PartialShape::dynamic(data_pshape.rank());
        }
    }

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, result_et, output_pshape);
}

shared_ptr<Node>
    op::v1::ConvolutionBackpropData::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 3)
    {
        return make_shared<v1::ConvolutionBackpropData>(new_args.at(0),
                                                        new_args.at(1),
                                                        new_args.at(2),
                                                        m_strides,
                                                        m_pads_begin,
                                                        m_pads_end,
                                                        m_dilations,
                                                        m_auto_pad,
                                                        m_output_padding);
    }
    else
    {
        return make_shared<v1::ConvolutionBackpropData>(new_args.at(0),
                                                        new_args.at(1),
                                                        m_strides,
                                                        m_pads_begin,
                                                        m_pads_end,
                                                        m_dilations,
                                                        m_auto_pad,
                                                        m_output_padding);
    }
}
