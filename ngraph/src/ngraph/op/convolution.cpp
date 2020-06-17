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
#include "ngraph/op/reverse.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "ngraph/runtime/reference/convolution.hpp"

using namespace std;
using namespace ngraph;

// *** Convolution OP SET 1 ***
constexpr NodeTypeInfo op::v1::Convolution::type_info;

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
        if (data_batch_shape.is_static() && filters_shape.is_static())
        {
            m_pads_begin.clear();
            m_pads_end.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            infer_auto_padding(data_batch_shape.to_shape(),
                               filter_shape,
                               m_strides,
                               m_dilations,
                               m_auto_pad,
                               m_pads_end,
                               m_pads_begin);
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

void op::v1::Convolution::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);
    const auto x_shape = x.get_shape();

    auto f = input_value(1);
    const auto f_shape = f.get_shape();

    adjoints.add_delta(x,
                       make_shared<op::v1::ConvolutionBackpropData>(
                           delta,
                           f,
                           op::Constant::create(element::i64, Shape{x_shape.size()}, x_shape),
                           m_strides,
                           m_pads_begin,
                           m_pads_end,
                           m_dilations,
                           m_auto_pad));

    adjoints.add_delta(f,
                       make_shared<op::v1::ConvolutionBackpropFilters>(
                           x,
                           delta,
                           op::Constant::create(element::i64, Shape{x_shape.size()}, f_shape),
                           m_strides,
                           m_dilations,
                           m_pads_begin,
                           m_pads_end));
}

constexpr NodeTypeInfo op::v1::ConvolutionBackpropData::type_info;
shared_ptr<Node> op::v1::Convolution::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const void *filter,
              const Shape& filter_shape, const Strides& strides, const Strides& dilation,
              const CoordinateDiff& pad_above, const CoordinateDiff& pad_below)
{
    auto out_data_ptr = out->get_data_ptr<ET>();
    auto in_data_ptr = arg0->get_data_ptr<ET>();
    auto out_shape = out->get_shape();
    auto in_shape = arg0->get_shape();
    const auto filter_data = reinterpret_cast<const typename element_type_traits<ET>::value_type *>(filter);
    Strides in_dilation(std::vector<size_t> (in_shape.size() - 2));
    std::fill(in_dilation.begin(), in_dilation.end(), 1);

    runtime::reference::convolution<typename element_type_traits<ET>::value_type>(in_data_ptr, filter_data, out_data_ptr,
            in_shape, filter_shape, out_shape, strides, dilation, pad_above, pad_below, in_dilation);
    return true;

}

bool evaluate_convolution(const HostTensorPtr& arg0, const HostTensorPtr& out, const void *filter,
                          const Shape& filter_shape, const Strides& strides, const Strides& dilation,
                          const CoordinateDiff& pad_above, const CoordinateDiff& pad_below)
{
    bool rc = true;
    switch (arg0->get_element_type())
    {
        case element::Type_t::undefined: rc = false; break;
        case element::Type_t::dynamic: rc = false; break;
        case element::Type_t::u1:
            rc = false;
            break;
// TODO: Arithmetic operators are not defined for bfloat16. Issue 33808
//        TYPE_CASE(bf16)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
//            break;
        TYPE_CASE(f16)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(f32)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(f64)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(i8)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(i16)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(i32)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(i64)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(u8)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(u16)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(u32)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(u64)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        TYPE_CASE(boolean)(arg0, out, filter, filter_shape, strides, dilation, pad_above, pad_below);
            break;
        default: rc = false; break;
    }
    return rc;

}
}

bool op::v1::Convolution::evaluate(const HostTensorVector &output_values, const HostTensorVector &input_values) {
    const auto filter = dynamic_pointer_cast<op::Constant>(input_value(1).get_node_shared_ptr());
    NGRAPH_CHECK(filter!=nullptr, "Failed to get Convolution filter values!");
    const auto strides = get_strides();
    evaluate_convolution(input_values[0], output_values[0], filter->get_data_ptr(), filter->get_shape(), get_strides(),
            get_dilations(), get_pads_begin(), get_pads_end());
    return true;
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
    if (get_inputs().size() == 3 && !is_dynamic)
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
    bool is_output_shape_present = get_inputs().size() == 3;
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

    bool is_output_shape_present = get_inputs().size() == 3;
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

void op::v1::ConvolutionBackpropData::generate_adjoints(autodiff::Adjoints& adjoints,
                                                        const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);
    const auto x_shape = x.get_shape();

    auto f = input_value(1);
    const auto f_shape = f.get_shape();

    auto data_conv = make_shared<op::v1::Convolution>(
        delta, f, m_strides, m_pads_begin, m_pads_end, m_dilations, m_auto_pad);

    adjoints.add_delta(x, data_conv);

    Strides strides = m_dilations;
    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;
    const Shape& filters_shape = get_input_shape(1);
    for (size_t i = 0; i < f_shape.size() - 2; i++)
    {
        ptrdiff_t pads_begin_backward =
            (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) - m_pads_begin[i];
        pads_begin.push_back(pads_begin_backward);

        ptrdiff_t pads_end_backward =
            (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) * m_dilations[i] +
            ((m_pads_begin[i] + (get_output_shape()[i].get_length() - 1) * m_strides[i] +
              m_pads_end[i] - (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) * m_dilations[i]) %
             m_strides[i]) -
            m_pads_end[i];

        pads_end.push_back(pads_end_backward -
                           (pads_begin_backward + (x_shape[i + 2] - 1) * m_strides[i] +
                            pads_end_backward - (f_shape[i + 2] - 1) * m_dilations[i]) %
                               m_strides[i]);
    }

    auto swap_NC = [](const Output<Node>& n) {
        AxisVector ax_order = ngraph::get_default_order(n.get_shape());
        ax_order[0] = 1;
        ax_order[1] = 0;

        auto new_shape = n.get_shape();
        new_shape[0] = n.get_shape()[1];
        new_shape[1] = n.get_shape()[0];

        return make_shared<op::Reshape>(n, ax_order, new_shape);
    };

    delta = swap_NC(delta);
    x = swap_NC(x);

    shared_ptr<Node> filter_deconv_bprop = make_shared<op::v1::Convolution>(
        x, delta, strides, pads_begin, pads_end, Strides(x.get_shape().size() - 2, 1), m_auto_pad);
    AxisSet axes;
    for (size_t i = 2; i < filter_deconv_bprop->get_shape().size(); ++i)
    {
        axes.insert(i);
    }
    filter_deconv_bprop = make_shared<ngraph::op::Reverse>(filter_deconv_bprop, axes);
    adjoints.add_delta(f, filter_deconv_bprop);
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

constexpr NodeTypeInfo op::v1::ConvolutionBackpropFilters::type_info;

op::v1::ConvolutionBackpropFilters::ConvolutionBackpropFilters(const Output<Node>& data_batch,
                                                               const Output<Node>& output_delta,
                                                               const Output<Node>& filters_shape,
                                                               const Strides& strides,
                                                               const Strides& dilations,
                                                               const CoordinateDiff& pads_begin,
                                                               const CoordinateDiff& pads_end)
    : Op({data_batch, output_delta, filters_shape})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
{
    constructor_validate_and_infer_types();
}

bool op::v1::ConvolutionBackpropFilters::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    return true;
}

const Shape op::v1::ConvolutionBackpropFilters::get_filters_shape() const
{
    Shape shape;
    if (auto const_op = as_type<op::Constant>(input_value(2).get_node()))
    {
        shape = const_op->get_shape_val();
    }
    return shape;
}

void op::v1::ConvolutionBackpropFilters::validate_and_infer_types()
{
    // Backprop to filters is itself convolution, with inputs/outputs/attributes transmogrified as
    // follows.
    //
    //                          Forward   Backward
    // "N" axis for data batch  0         1
    // "C" axis for data batch  1         0
    // "Co" axis for filters    0         0
    // "Ci" axis for filters    1         1
    // "N" axis for output      0         1
    // "C" axis for output      1         0
    // Data batch               x         x
    // Data batch shape         S_x       S_x
    // Filters                  f         delta
    // Filters shape            S_f       S_f
    // Window movement strides  q_x       p_f
    // Window dilation strides  p_f       q_x
    // Padding below            a_x       a_x
    // Padding above            b_x       b_x - (a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q_x
    // Output shape             S_o       S_f
    //
    // To _validate_, we simply need to check/infer the output shape of the forward convolution,
    // then check to make sure that the incoming delta has the same shape as the forward output.
    //
    // We will also compute and store the various parameters in the "backward" column above, since
    // some backends need them. (TODO(amprocte): Is it just because of the way the reference works
    // that this stuff is needed? If so, we can probably get rid of it and have conv_backprop
    // reference kernels that do the calculations of the backward parameters internally, or supply
    // utility functions to do it.)

    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& delta_shape = get_input_partial_shape(1);
    element::Type delta_et = get_input_element_type(1);

    element::Type forward_result_et;
    PartialShape forward_result_shape;
    PartialShape filters_shape{PartialShape::dynamic()};

    if (input_value(2).get_node_shared_ptr()->is_constant())
    {
        filters_shape = get_filters_shape();
    }

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(forward_result_et, data_batch_et, delta_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        delta_et,
        ").");

    if (input_value(1).get_node_shared_ptr()->is_constant())
    {
        forward_result_shape =
            infer_convolution_forward(this,
                                      data_batch_shape,
                                      Strides(data_batch_shape.rank().get_length() - 2, 1),
                                      m_pads_begin,
                                      m_pads_end,
                                      filters_shape,
                                      m_strides,
                                      m_dilations);

        NODE_VALIDATION_CHECK(this,
                              forward_result_shape.compatible(delta_shape),
                              "Inferred forward output shape (",
                              forward_result_shape,
                              ") does not match shape of ",
                              "delta (",
                              delta_shape,
                              ").");
    }

    set_input_is_relevant_to_shape(2);
    set_output_type(0, forward_result_et, filters_shape);
}

shared_ptr<Node>
    op::v1::ConvolutionBackpropFilters::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::ConvolutionBackpropFilters>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       m_strides,
                                                       m_dilations,
                                                       m_pads_begin,
                                                       m_pads_end);
}

CoordinateDiff op::v1::ConvolutionBackpropFilters::compute_backward_in_pad_above() const
{
    const auto& in_shape = get_input_shape(0);
    const auto& out_shape = get_input_shape(1);
    const auto& filter_shape = get_filters_shape();
    const auto& in_pad_above = get_pads_end();
    const auto& in_pad_below = get_pads_begin();
    const auto& filter_dilation = get_dilations();
    const auto& stride = get_strides();
    size_t spatial_dim_count = static_cast<size_t>(out_shape.size()) - 2;
    CoordinateDiff backward_in_pad_above;
    backward_in_pad_above.resize(spatial_dim_count);

    for (size_t i = 0; i < spatial_dim_count; i++)
    {
        backward_in_pad_above[i] =
            in_pad_above[i] -
            (in_pad_below[i] + (static_cast<ptrdiff_t>(in_shape[i + 2]) - 1) + in_pad_above[i] -
             (filter_shape[i + 2] - 1) * filter_dilation[i]) %
                stride[i];
    }
    return backward_in_pad_above;
}

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

void op::v0::Convolution::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);
    const auto x_shape = x.get_shape();

    auto f = input_value(1);
    const auto f_shape = f.get_shape();

    adjoints.add_delta(x,
                       make_shared<op::v0::ConvolutionBackpropData>(x_shape,
                                                                    f,
                                                                    delta,
                                                                    m_window_movement_strides,
                                                                    m_window_dilation_strides,
                                                                    m_padding_below,
                                                                    m_padding_above,
                                                                    m_data_dilation_strides));

    adjoints.add_delta(f,
                       make_shared<op::v0::ConvolutionBackpropFilters>(x,
                                                                       f_shape,
                                                                       delta,
                                                                       m_window_movement_strides,
                                                                       m_window_dilation_strides,
                                                                       m_padding_below,
                                                                       m_padding_above,
                                                                       m_data_dilation_strides));
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

void op::v0::ConvolutionBackpropData::generate_adjoints(autodiff::Adjoints& adjoints,
                                                        const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(1);
    const auto x_shape = x.get_shape();

    auto f = input_value(0);
    const auto f_shape = f.get_shape();

    auto data_conv = make_shared<op::v0::Convolution>(delta,
                                                      f,
                                                      m_window_movement_strides_forward,
                                                      m_window_dilation_strides_forward,
                                                      m_padding_below_forward,
                                                      m_padding_above_forward,
                                                      m_data_dilation_strides_forward);

    adjoints.add_delta(x, data_conv);

    Strides window_movement_strides = m_window_dilation_strides_forward;
    Strides window_dilation_strides = m_data_dilation_strides_forward;
    Strides data_dilation_strides = m_window_movement_strides_forward;
    CoordinateDiff padding_below;
    CoordinateDiff padding_above;
    const Shape& filters_shape = get_input_shape(0);
    for (size_t i = 0; i < f_shape.size() - 2; i++)
    {
        ptrdiff_t padding_below_backward =
            (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) * window_dilation_strides[i] -
            m_padding_below_forward[i];
        padding_below.push_back(padding_below_backward);

        ptrdiff_t padding_above_backward =
            (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                m_window_dilation_strides_forward[i] +
            ((m_padding_below_forward[i] +
              ((m_data_batch_shape[i + 2]) - 1) * m_data_dilation_strides_forward[i] +
              m_padding_above_forward[i] -
              (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                  m_window_dilation_strides_forward[i]) %
             m_window_movement_strides_forward[i]) -
            m_padding_above_forward[i];

        padding_above.push_back(
            padding_above_backward -
            (padding_below_backward + (x_shape[i + 2] - 1) * m_window_movement_strides_forward[i] +
             padding_above_backward - (f_shape[i + 2] - 1) * m_window_dilation_strides_forward[i]) %
                m_data_dilation_strides_forward[i]);
    }

    auto swap_NC = [](const Output<Node>& n) {
        AxisVector ax_order = ngraph::get_default_order(n.get_shape());
        ax_order[0] = 1;
        ax_order[1] = 0;

        auto new_shape = n.get_shape();
        new_shape[0] = n.get_shape()[1];
        new_shape[1] = n.get_shape()[0];

        return make_shared<op::Reshape>(n, ax_order, new_shape);
    };

    delta = swap_NC(delta);
    x = swap_NC(x);

    shared_ptr<Node> filter_deconv_bprop = make_shared<op::v0::Convolution>(x,
                                                                            delta,
                                                                            window_movement_strides,
                                                                            window_dilation_strides,
                                                                            padding_below,
                                                                            padding_above,
                                                                            data_dilation_strides);
    AxisSet axes;
    for (size_t i = 2; i < filter_deconv_bprop->get_shape().size(); ++i)
    {
        axes.insert(i);
    }
    filter_deconv_bprop = make_shared<ngraph::op::Reverse>(filter_deconv_bprop, axes);
    adjoints.add_delta(f, filter_deconv_bprop);
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

constexpr NodeTypeInfo op::v0::ConvolutionBackpropFilters::type_info;

op::v0::ConvolutionBackpropFilters::ConvolutionBackpropFilters(
    const Output<Node>& data_batch,
    const Shape& filters_shape,
    const Output<Node>& output_delta,
    const Strides& window_movement_strides_forward,
    const Strides& window_dilation_strides_forward,
    const CoordinateDiff& padding_below_forward,
    const CoordinateDiff& padding_above_forward,
    const Strides& data_dilation_strides_forward)
    : Op({data_batch, output_delta})
    , m_filters_shape(filters_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    constructor_validate_and_infer_types();
}

bool op::v0::ConvolutionBackpropFilters::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("m_filters_shape", m_filters_shape);
    visitor.on_attribute("window_movement_strides_forward", m_window_movement_strides_forward);
    visitor.on_attribute("window_dilation_strides_forward", m_window_dilation_strides_forward);
    visitor.on_attribute("padding_below_forward", m_padding_below_forward);
    visitor.on_attribute("padding_above_forward", m_padding_above_forward);
    visitor.on_attribute("data_dilation_strides_forward", m_data_dilation_strides_forward);
    return true;
}

void op::v0::ConvolutionBackpropFilters::validate_and_infer_types()
{
    // Backprop to filters is itself convolution, with inputs/outputs/attributes transmogrified as
    // follows.
    //
    //                          Forward   Backward
    // "N" axis for data batch  0         1
    // "C" axis for data batch  1         0
    // "Co" axis for filters    0         0
    // "Ci" axis for filters    1         1
    // "N" axis for output      0         1
    // "C" axis for output      1         0
    // Data batch               x         x
    // Data batch shape         S_x       S_x
    // Filters                  f         delta
    // Filters shape            S_f       S_f
    // Window movement strides  q_x       p_f
    // Window dilation strides  p_f       q_x
    // Padding below            a_x       a_x
    // Padding above            b_x       b_x - (a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q_x
    // Data dilation strides    p_x       p_x
    // Output shape             S_o       S_f
    //
    // To _validate_, we simply need to check/infer the output shape of the forward convolution,
    // then check to make sure that the incoming delta has the same shape as the forward output.
    //
    // We will also compute and store the various parameters in the "backward" column above, since
    // some backends need them. (TODO(amprocte): Is it just because of the way the reference works
    // that this stuff is needed? If so, we can probably get rid of it and have conv_backprop
    // reference kernels that do the calculations of the backward parameters internally, or supply
    // utility functions to do it.)

    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& delta_shape = get_input_partial_shape(1);
    element::Type delta_et = get_input_element_type(1);

    element::Type forward_result_et;
    PartialShape forward_result_shape;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(forward_result_et, data_batch_et, delta_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        delta_et,
        ").");

    forward_result_shape = infer_convolution_forward(this,
                                                     data_batch_shape,
                                                     m_data_dilation_strides_forward,
                                                     m_padding_below_forward,
                                                     m_padding_above_forward,
                                                     m_filters_shape,
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

    set_output_type(0, forward_result_et, m_filters_shape);
}

shared_ptr<Node>
    op::v0::ConvolutionBackpropFilters::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::ConvolutionBackpropFilters>(new_args.at(0),
                                                       m_filters_shape,
                                                       new_args.at(1),
                                                       m_window_movement_strides_forward,
                                                       m_window_dilation_strides_forward,
                                                       m_padding_below_forward,
                                                       m_padding_above_forward,
                                                       m_data_dilation_strides_forward);
}

CoordinateDiff op::v0::ConvolutionBackpropFilters::compute_backward_in_pad_above() const
{
    const auto& in_shape = get_input_shape(0);
    const auto& out_shape = get_input_shape(1);
    const auto& filter_shape = get_filters_shape();
    const auto& in_pad_above = get_padding_above_forward();
    const auto& in_pad_below = get_padding_below_forward();
    const auto& in_dilation = get_data_dilation_strides_forward();
    const auto& filter_dilation = get_window_dilation_strides_forward();
    const auto& stride = get_window_movement_strides_forward();
    size_t spatial_dim_count = static_cast<size_t>(out_shape.size()) - 2;
    CoordinateDiff backward_in_pad_above;
    backward_in_pad_above.resize(spatial_dim_count);

    for (size_t i = 0; i < spatial_dim_count; i++)
    {
        backward_in_pad_above[i] =
            in_pad_above[i] -
            (in_pad_below[i] + (static_cast<ptrdiff_t>(in_shape[i + 2]) - 1) * in_dilation[i] +
             in_pad_above[i] - (filter_shape[i + 2] - 1) * filter_dilation[i]) %
                stride[i];
    }
    return backward_in_pad_above;
}

//
// This is a legacy function, retained because the CPU backend uses it for now.
// TODO(amprocte): Update CPU backend to use the new stuff in validation_util.hpp, and remove this
// function.
//
Shape op::util::infer_convolution_output_shape(const Node* node,
                                               const Shape& data_batch_shape,
                                               const Shape& filters_shape,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               size_t batch_axis_data,
                                               size_t input_channel_axis_data,
                                               size_t input_channel_axis_filters,
                                               size_t output_channel_axis_filters,
                                               size_t batch_axis_result,
                                               size_t output_channel_axis_result)
{
    NODE_VALIDATION_CHECK(node, batch_axis_data <= 1, "(This is an internal nGraph error)");
    NODE_VALIDATION_CHECK(node, input_channel_axis_data <= 1, "(This is an internal nGraph error)");
    NODE_VALIDATION_CHECK(
        node, input_channel_axis_filters <= 1, "(This is an internal nGraph error)");
    NODE_VALIDATION_CHECK(
        node, output_channel_axis_filters <= 1, "(This is an internal nGraph error)");
    NODE_VALIDATION_CHECK(node, batch_axis_result <= 1, "(This is an internal nGraph error)");
    NODE_VALIDATION_CHECK(
        node, output_channel_axis_result <= 1, "(This is an internal nGraph error)");

    //
    // Make sure data_batch: NCiDi for some Di of rank>0, N != 0, Ci != 0.
    //
    NODE_VALIDATION_CHECK(node,
                          data_batch_shape.size() >= 3,
                          "Data batch input must have rank of at least 3 (one batch axis, ",
                          "one input-channel axis, and at least one spatial dimension) ",
                          "(data batch shape: ",
                          data_batch_shape,
                          ").");

    size_t batch_size = data_batch_shape[batch_axis_data];
    NODE_VALIDATION_CHECK(node,
                          batch_size != 0,
                          "Data batch size is zero (data batch shape: ",
                          data_batch_shape,
                          ", ",
                          "batch axis is axis ",
                          batch_axis_data,
                          ").");

    size_t input_channel_count = data_batch_shape[input_channel_axis_data];
    NODE_VALIDATION_CHECK(node,
                          input_channel_count != 0,
                          "Input channel count is zero (data batch shape: ",
                          data_batch_shape,
                          ", ",
                          "channel axis is axis ",
                          input_channel_axis_data,
                          ").");

    size_t spatial_dimension_count = data_batch_shape.size() - 2;

    //
    // Make sure filters: CoCiWv for some Co>0, rank of W = rank of Di.
    //
    NODE_VALIDATION_CHECK(
        node,
        filters_shape.size() == 2 + spatial_dimension_count,
        "Filter input must have rank equal to the data batch (one axis for output ",
        "channels, one axis for input channels, and the same number of spatial ",
        "dimensions as the data batch (filter input shape: ",
        filters_shape,
        ", ",
        "data batch shape: ",
        data_batch_shape,
        ").");

    size_t output_channel_count = filters_shape[output_channel_axis_filters];
    NODE_VALIDATION_CHECK(node,
                          output_channel_count != 0,
                          "Output channel count for filters is zero (filters shape: ",
                          filters_shape,
                          ", ",
                          "output channels on axis ",
                          output_channel_axis_filters,
                          ").");

    NODE_VALIDATION_CHECK(node,
                          filters_shape[input_channel_axis_filters] == input_channel_count,
                          "Input channel count for filters (",
                          filters_shape[input_channel_axis_filters],
                          ") ",
                          "does not match the number of channels in the data batch (",
                          input_channel_count,
                          ") ",
                          "(filter input shape: ",
                          filters_shape,
                          ", filter input channels on axis ",
                          input_channel_axis_filters,
                          "; data batch shape: ",
                          data_batch_shape,
                          ", data batch channels on axis ",
                          batch_axis_data,
                          ").");

    //
    // Make sure window movement strides, window dilation strides, and data dilation strides
    // have same rank as Di.
    //
    NODE_VALIDATION_CHECK(
        node,
        window_movement_strides.size() == spatial_dimension_count,
        "Rank of window movement strides does not match the number of spatial dimensions (",
        spatial_dimension_count,
        ") in the data batch (window movement strides: ",
        window_movement_strides,
        ", data batch shape: ",
        data_batch_shape,
        ").");

    NODE_VALIDATION_CHECK(
        node,
        window_dilation_strides.size() == spatial_dimension_count,
        "Rank of window dilation strides does not match the number of spatial dimensions (",
        spatial_dimension_count,
        ") in the data batch (window dilation strides: ",
        window_dilation_strides,
        ", data batch shape: ",
        data_batch_shape,
        ").");

    NODE_VALIDATION_CHECK(
        node,
        data_dilation_strides.size() == spatial_dimension_count,
        "Rank of data dilation strides does not match the number of spatial dimensions (",
        spatial_dimension_count,
        ") in the data batch (data dilation strides: ",
        data_dilation_strides,
        ", data batch shape: ",
        data_batch_shape,
        ").");

    //
    // Make sure padding-below and padding-above shapes have same rank as Di.
    //
    NODE_VALIDATION_CHECK(
        node,
        padding_below.size() == spatial_dimension_count,
        "Rank of the padding below does not match the number of spatial dimensions (",
        spatial_dimension_count,
        ") in the data batch (padding below: ",
        padding_below,
        ", data batch shape: ",
        data_batch_shape,
        ").");

    NODE_VALIDATION_CHECK(
        node,
        padding_above.size() == spatial_dimension_count,
        "Rank of the padding above does not match the number of spatial dimensions (",
        spatial_dimension_count,
        ") in the data batch (padding above: ",
        padding_above,
        ", data batch shape: ",
        data_batch_shape,
        ").");

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0 after padding and
    // dilation.
    //
    std::vector<ptrdiff_t> input_item_virtual_shape_signed;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_CHECK(node,
                              data_dilation_strides[i] != 0,
                              "Data dilation stride at spatial dimension ",
                              i,
                              " is zero ",
                              "(data dilation strides: ",
                              data_dilation_strides,
                              ").");

        size_t dim_size = data_batch_shape[1 + 1 + i];
        size_t dilated_dim_size = (dim_size - 1) * data_dilation_strides[i] + 1;

        ptrdiff_t padded_dilated_dim_size = padding_below[i] + dilated_dim_size + padding_above[i];

        input_item_virtual_shape_signed.push_back(padded_dilated_dim_size);
    }

    Shape input_item_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_CHECK(node,
                              input_item_virtual_shape_signed[i] > 0,
                              "Input dimension after padding and dilation is non-positive ",
                              "at spatial axis ",
                              i,
                              " (post-padding/dilation input item shape: ",
                              input_item_virtual_shape,
                              ", data batch shape: ",
                              data_batch_shape,
                              ", data dilation strides: ",
                              data_dilation_strides,
                              ", padding below: ",
                              padding_below,
                              ", padding above: ",
                              padding_above,
                              ").");

        input_item_virtual_shape.push_back(size_t(input_item_virtual_shape_signed[i]));
    }

    //
    // Extract the physical shape Wp of the convolution window, *not* including dilation, from the
    // filter dimensions. At the same time, make sure window shape dimensions are all larger than
    // 0.
    //
    Shape window_physical_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        window_physical_shape.push_back(filters_shape[1 + 1 + i]);
        NODE_VALIDATION_CHECK(node,
                              window_physical_shape[i] != 0,
                              "Filters shape at spatial dimension ",
                              i,
                              " is zero ",
                              "(filters shape: ",
                              filters_shape,
                              ").");
    }

    //
    // Compute virtual shape Wp of the convolution window, *including* dilation. At the same time,
    // make sure all window dilation strides are larger than 0, and that the dilated filter fits
    // within the spatial dimensions.
    //
    Shape window_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_CHECK(node,
                              window_dilation_strides[i] != 0,
                              "Window dilation stride at spatial dimension ",
                              i,
                              " is zero ",
                              "(window dilation strides: ",
                              window_dilation_strides,
                              ").");

        window_virtual_shape.push_back((window_physical_shape[i] - 1) * window_dilation_strides[i] +
                                       1);

        NODE_VALIDATION_CHECK(
            node,
            window_virtual_shape[i] <= input_item_virtual_shape[i],
            "Post-dilation window shape is smaller than the post-padding/dilation ",
            "input item shape at spatial dimension ",
            i,
            " (post-padding/dilation ",
            "input item shape: ",
            input_item_virtual_shape,
            ", data batch shape: ",
            data_batch_shape,
            ", data dilation strides: ",
            data_dilation_strides,
            ", padding below: ",
            padding_below,
            ", padding above: ",
            padding_above,
            ", post-dilation window shape: ",
            window_virtual_shape,
            ", filters shape: ",
            filters_shape,
            ", window dilation strides: ",
            window_dilation_strides);
    }

    //
    // Construct result shape: NCoDo or CoNDo (depending on *_axis_result), checking at the same
    // time that all window movement strides are larger than 0.
    //
    Shape result_shape(spatial_dimension_count + 2);
    result_shape[batch_axis_result] = batch_size;
    result_shape[output_channel_axis_result] = output_channel_count;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_CHECK(node,
                              window_movement_strides[i] != 0,
                              "Window movement stride at spatial dimension ",
                              i,
                              " is zero ",
                              "(window movement strides: ",
                              window_movement_strides,
                              ").");

        result_shape[i + 2] = ceil_div(input_item_virtual_shape[i] - window_virtual_shape[i] + 1,
                                       window_movement_strides[i]);
    }

    return result_shape;
}
