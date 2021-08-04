// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/convolution.hpp"
#include "itt.hpp"
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
    NGRAPH_OP_SCOPE(v1_Convolution_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

int64_t calculate_num_spatial_dims_and_update_attributes(op::v1::Convolution* node,
                                                         PartialShape& input_shape,
                                                         PartialShape& filters_shape,
                                                         Strides& dilations,
                                                         Strides& strides,
                                                         CoordinateDiff& pad_begin,
                                                         CoordinateDiff& pad_end,
                                                         const op::PadType& auto_pad)
{
    int64_t num_spatial_dims = -1;
    if (const auto& size = dilations.size())
        num_spatial_dims = static_cast<int64_t>(size);
    if (const auto& size = strides.size())
        num_spatial_dims = static_cast<int64_t>(size);
    if (const auto& size = pad_begin.size())
        num_spatial_dims = static_cast<int64_t>(size);
    if (const auto& size = pad_end.size())
        num_spatial_dims = static_cast<int64_t>(size);
    const auto& input_rank = input_shape.rank();
    if (input_rank.is_static())
        num_spatial_dims = input_rank.get_length() - 2;
    const auto& filters_rank = filters_shape.rank();
    if (filters_rank.is_static())
        num_spatial_dims = filters_rank.get_length() - 2;

    if (num_spatial_dims == -1)
        return num_spatial_dims; // can not deduce output rank

    if (strides.empty())
    {
        strides = Strides(num_spatial_dims, 1);
        node->set_strides(strides);
    }
    if (dilations.empty())
    {
        dilations = Strides(num_spatial_dims, 1);
        node->set_dilations(dilations);
    }
    if (pad_begin.empty() || auto_pad == op::PadType::VALID)
    {
        pad_begin = CoordinateDiff(num_spatial_dims, 0);
        node->set_pads_begin(pad_begin);
    }
    if (pad_end.empty() || auto_pad == op::PadType::VALID)
    {
        pad_end = CoordinateDiff(num_spatial_dims, 0);
        node->set_adding_above(pad_end);
    }


    NODE_VALIDATION_CHECK(node,
                          strides.size() == num_spatial_dims,
                          "Strides should be defined for all and only spatial features.");
    NODE_VALIDATION_CHECK(node,
                          dilations.size() == num_spatial_dims,
                          "Dilations should be defined for all and only spatial features.");
    NODE_VALIDATION_CHECK(node,
                          pad_begin.size() == num_spatial_dims &&
                              pad_end.size() == num_spatial_dims,
                          "Pads should be defined for all and only spatial features.");
    NODE_VALIDATION_CHECK(
        node,
        (input_rank.is_dynamic() || input_rank.get_length() == num_spatial_dims + 2) &&
            (filters_rank.is_dynamic() || filters_rank.get_length() == num_spatial_dims + 2),
        "Data batch and filters rank do not match (data batch shape: ",
        input_shape,
        ", filters shape: ",
        filters_shape,
        ").");

    if (input_rank.is_dynamic())
        input_shape = PartialShape::dynamic(num_spatial_dims + 2);
    if (filters_rank.is_dynamic())
        filters_shape = PartialShape::dynamic(num_spatial_dims + 2);

    NODE_VALIDATION_CHECK(
        node,
        std::all_of(dilations.begin(), dilations.end(), [](const size_t& i) { return i > 0; }),
        "Filter dilation (",
        dilations,
        ") has zero dimension.");
    NODE_VALIDATION_CHECK(
        node,
        std::all_of(strides.begin(), strides.end(), [](const size_t& i) { return i > 0; }),
        "Filter strides (",
        strides,
        ") has zero dimension.");
    return num_spatial_dims;
}

void convolution_shape_infer(op::v1::Convolution* node,
                             PartialShape& input_shape,
                             PartialShape& filters_shape,
                             PartialShape& output_shape)
{
    auto dilations = node->get_dilations();
    auto strides = node->get_strides();
    auto pad_begin = node->get_pads_begin(), pad_end = node->get_pads_end();
    const auto& auto_pad = node->get_auto_pad();

    int64_t num_spatial_dims = calculate_num_spatial_dims_and_update_attributes(
        node, input_shape, filters_shape, dilations, strides, pad_begin, pad_end, auto_pad);
    if (num_spatial_dims < 1)
        return;
    // ranks are originally static or aligned with num_spatial_dims, attributes are valid

    output_shape = PartialShape::dynamic(num_spatial_dims + 2);
    output_shape[0] = input_shape[0];
    output_shape[1] = filters_shape[0];

    NODE_VALIDATION_CHECK(node,
                          input_shape[1].is_dynamic() || filters_shape.is_dynamic() ||
                              input_shape[1] == filters_shape[1],
                          "Data batch channel count (",
                          input_shape[1],
                          ") does not match filter input ",
                          "channel count (",
                          filters_shape[1],
                          ").");

    for (size_t i = 0; i < num_spatial_dims; ++i)
    {
        const auto& input_dim = input_shape[i + 2];
        const auto& filters_dim = filters_shape[i + 2];
        if (input_dim.is_static() && filters_dim.is_static())
        {
            const int64_t& window_dilated_dim = (filters_dim.get_length() - 1) * dilations[i] + 1;
            NODE_VALIDATION_CHECK(node,
                                  window_dilated_dim > 0,
                                  "Window after dilation has dimension less than 1 (dim: ",
                                  window_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");
            if (auto_pad == op::PadType::SAME_UPPER || auto_pad == op::PadType::SAME_LOWER)
            {
                const int64_t& image_size = input_dim.get_length();
                const int64_t& filter_stride = strides[i];
                const int64_t& output_size = (image_size + filter_stride - 1) / filter_stride;

                const int64_t& tmp = (output_size - 1) * filter_stride + window_dilated_dim;
                const int64_t& padding_needed = tmp > image_size ? tmp - image_size : 0;

                const size_t& padding_lhs = static_cast<size_t>(padding_needed / 2);
                const size_t& padding_rhs = static_cast<size_t>(padding_needed - padding_lhs);

                pad_begin[i] = auto_pad == op::PadType::SAME_UPPER ? padding_lhs : padding_rhs;
                pad_end[i] = auto_pad == op::PadType::SAME_UPPER ? padding_rhs : padding_lhs;
            }

            const int64_t& data_padded_dilated_dim =
                input_dim.get_length() + pad_begin[i] + pad_end[i];
            NODE_VALIDATION_CHECK(node,
                                  window_dilated_dim <= data_padded_dilated_dim,
                                  "Window after dilation has dimension (dim: ",
                                  window_dilated_dim,
                                  ") larger than the data shape after padding (dim: ",
                                  data_padded_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");
            output_shape[i + 2] = (data_padded_dilated_dim - window_dilated_dim) / strides[i] + 1;
        }
    }
    node->set_pads_begin(pad_begin);
    node->set_adding_above(pad_end);
}

void op::v1::Convolution::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Convolution_validate_and_infer_types);
    element::Type data_batch_et = get_input_element_type(0);
    element::Type filters_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_batch_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        filters_et,
        ").");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element types must be numeric. Got: ",
                          result_et);

    PartialShape input_shape = get_input_partial_shape(0);
    PartialShape filters_shape = get_input_partial_shape(1);
    PartialShape result_shape;
    convolution_shape_infer(this, input_shape, filters_shape, result_shape);
    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::v1::Convolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Convolution_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Convolution>(new_args.at(0),
                                        new_args.at(1),
                                        m_strides,
                                        m_pads_begin,
                                        m_pads_end,
                                        m_dilations,
                                        m_auto_pad);
}

shared_ptr<Node> op::v1::Convolution::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

// *** ConvolutionBackpropData OP SET 1 ***
NGRAPH_RTTI_DEFINITION(op::v1::ConvolutionBackpropData, "ConvolutionBackpropData", 1);

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
    NGRAPH_OP_SCOPE(v1_ConvolutionBackpropData_visit_attributes);
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
        return !has_and_set_equal_bounds(input_value(2));
    }
    return is_dynamic;
}

const PartialShape op::v1::ConvolutionBackpropData::get_output_shape() const
{
    auto data_pshape = get_input_partial_shape(0);
    auto filter_pshape = get_input_partial_shape(1);

    PartialShape shape;
    bool is_output_shape_present = inputs().size() == 3;
    if (is_output_shape_present)
    {
        if (const auto& const_op = get_constant_from_source(input_value(2)))
        {
            return PartialShape{const_op->get_shape_val()};
        }
    }

    if (data_pshape.rank().is_static())
    {
        shape = PartialShape{vector<Dimension>(data_pshape.rank().get_length() - 2)};
    }
    else if (filter_pshape.rank().is_static())
    {
        shape = PartialShape{vector<Dimension>(filter_pshape.rank().get_length() - 2)};
    }
    else
    {
        shape = PartialShape::dynamic();
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
    NGRAPH_OP_SCOPE(v1_ConvolutionBackpropData_validate_and_infer_types);
    const PartialShape& data_pshape = get_input_partial_shape(0);
    element::Type delta_et = get_input_element_type(0);
    const PartialShape& filters_pshape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, delta_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        delta_et,
        ", filters element type: ",
        filters_et,
        ").");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    Rank result_ps_rank;
    NODE_VALIDATION_CHECK(this,
                          Rank::merge(result_ps_rank, data_pshape.rank(), filters_pshape.rank()),
                          "Data and filters inputs must have same rank. Got: ",
                          data_pshape,
                          " and ",
                          filters_pshape);

    NODE_VALIDATION_CHECK(this,
                          result_ps_rank.compatible(3) || result_ps_rank.compatible(4) ||
                              result_ps_rank.compatible(5),
                          "Data and filters inputs must have rank 3, 4 or 5. Got: ",
                          result_ps_rank);

    if (data_pshape.rank().is_static() && filters_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(
            this,
            data_pshape[1].compatible(filters_pshape[0]),
            "Input channels dimension of data and filters inputs must be equal. Got: ",
            data_pshape,
            " and ",
            filters_pshape);
    }

    bool is_output_shape_present = inputs().size() == 3;
    if (is_output_shape_present)
    {
        const PartialShape& output_shape_pshape = get_input_partial_shape(2);
        const element::Type output_shape_et = get_input_element_type(2);

        NODE_VALIDATION_CHECK(this,
                              output_shape_et.is_integral_number(),
                              "Element type for output shape should be of integer type ",
                              "(output_shape element type: ",
                              output_shape_et,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              output_shape_pshape.rank().compatible(1),
                              "Spatial shape of output input must be of rank 1 ",
                              "(output_shape shape: ",
                              output_shape_pshape,
                              ").");
    }
    PartialShape output_spatial_pshape = get_output_shape();

    if (result_ps_rank.is_static())
    {
        const auto num_spatial_dims = result_ps_rank.get_length() - 2;
        if (m_strides.size() == 0)
        {
            m_strides = Strides(num_spatial_dims, 1);
        }
        if (m_dilations.size() == 0)
        {
            m_dilations = Strides(num_spatial_dims, 1);
        }
        if (m_pads_begin.size() == 0 || m_auto_pad == PadType::VALID)
        {
            m_pads_begin = CoordinateDiff(num_spatial_dims, 0);
        }
        if (m_pads_end.size() == 0 || m_auto_pad == PadType::VALID)
        {
            m_pads_end = CoordinateDiff(num_spatial_dims, 0);
        }
        if (m_output_padding.size() == 0)
        {
            m_output_padding = CoordinateDiff(num_spatial_dims, 0);
        }

        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_strides.size()) == num_spatial_dims,
                              "Strides should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_dilations.size()) == num_spatial_dims,
                              "Dilations should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_begin.size()) == num_spatial_dims &&
                                  static_cast<int64_t>(m_pads_end.size()) == num_spatial_dims,
                              "Pads should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_output_padding.size()) == num_spatial_dims,
                              "Output padding should be defined for all and only "
                              "spatial features.");

        if (is_output_shape_present && output_spatial_pshape.is_static())
        {
            Shape output_shape = output_spatial_pshape.to_shape();
            NODE_VALIDATION_CHECK(this,
                                  static_cast<int64_t>(output_shape.size()) == num_spatial_dims,
                                  "Output shape should be specified only and for "
                                  "all spatial dimensions.");
        }
    }

    PartialShape result_pshape{PartialShape::dynamic()};
    // If output shape is provided, ignore current values for padding begin/end
    // and infer them.
    if (is_output_shape_present)
    {
        if (output_spatial_pshape.rank().is_static())
        {
            if (data_pshape.rank().is_static() && filters_pshape.rank().is_static())
            {
                const PartialShape data_spatial_shape = [data_pshape]() {
                    vector<Dimension> data_dims{data_pshape};
                    data_dims.erase(data_dims.begin(), data_dims.begin() + 2); // remove {N, C_IN}
                    return PartialShape{data_dims};
                }();

                const PartialShape filters_spatial_shape = [filters_pshape]() {
                    vector<Dimension> filters_dims{filters_pshape};
                    filters_dims.erase(filters_dims.begin(),
                                       filters_dims.begin() + 2); // remove {C_IN, C_OUT}
                    return PartialShape{filters_dims};
                }();

                // If auto_pad has one of following mode we infer paddings. Otherwise in
                // EXPLICIT auto_pad mode we use what is provided.
                if ((m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER) &&
                    (data_spatial_shape.is_static() && filters_spatial_shape.is_static() &&
                     output_spatial_pshape.is_static()))
                {
                    opset1::infer_conv_backprop_auto_padding(data_spatial_shape.to_shape(),
                                                             filters_spatial_shape.to_shape(),
                                                             output_spatial_pshape.to_shape(),
                                                             m_strides,
                                                             m_dilations,
                                                             m_auto_pad,
                                                             m_output_padding,
                                                             m_pads_begin,
                                                             m_pads_end);
                }
            }
            vector<Dimension> output_pshape{output_spatial_pshape};
            // C_OUT
            auto n_out_channels =
                filters_pshape.rank().is_static() ? filters_pshape[1] : Dimension::dynamic();
            output_pshape.insert(output_pshape.begin(), n_out_channels);
            // N
            auto batches = data_pshape.rank().is_static() ? data_pshape[0] : Dimension::dynamic();
            output_pshape.insert(output_pshape.begin(), batches);
            result_pshape = PartialShape{output_pshape};
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

        vector<Dimension> output_pshape;
        if (data_pshape.rank().is_static() && filters_pshape.rank().is_static())
        {
            auto data_spatial_shape = [data_pshape]() {
                vector<Dimension> data_dims{data_pshape};
                return vector<Dimension>{std::next(data_dims.begin(), 2),
                                         std::end(data_dims)}; // remove {N, C_IN}
            }();

            auto filters_spatial_shape = [filters_pshape]() {
                vector<Dimension> filters_dims{filters_pshape};
                return vector<Dimension>{std::next(filters_dims.begin(), 2), // remove {C_IN, C_OUT}
                                         std::end(filters_dims)};
            }();

            infer_conv_backprop_output_spatial_shape(data_spatial_shape,
                                                     filters_spatial_shape,
                                                     m_strides,
                                                     m_dilations,
                                                     m_pads_begin,
                                                     m_pads_end,
                                                     m_output_padding,
                                                     output_pshape);
        }
        else
        {
            output_pshape = vector<Dimension>{output_spatial_pshape};
        }

        if (output_pshape.size())
        {
            // C_OUT
            auto n_out_channels =
                filters_pshape.rank().is_static() ? filters_pshape[1] : Dimension::dynamic();
            output_pshape.insert(output_pshape.begin(), n_out_channels);
            // N
            auto batches = data_pshape.rank().is_static() ? data_pshape[0] : Dimension::dynamic();
            output_pshape.insert(output_pshape.begin(), batches);
            result_pshape = PartialShape{output_pshape};
        }
    }
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, result_et, result_pshape);
}

shared_ptr<Node>
    op::v1::ConvolutionBackpropData::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_ConvolutionBackpropData_clone_with_new_inputs);
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
