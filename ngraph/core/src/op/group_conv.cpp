// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include "itt.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

//------------------------------------------------------------------------------
//                        v1::GroupConvolution
//------------------------------------------------------------------------------

NGRAPH_RTTI_DEFINITION(op::v1::GroupConvolution, "GroupConvolution", 1);

shared_ptr<Node> op::v1::GroupConvolution::get_default_value() const
{
    return op::Constant::create(get_element_type(), get_shape(), {0});
}

op::v1::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
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

bool ngraph::op::v1::GroupConvolution::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_GroupConvolution_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

static Dimension infer_group_from_input_shapes(const PartialShape& data_pshape,
                                               const PartialShape& filters_pshape)
{
    Dimension group_dim = Dimension();
    if (data_pshape.rank().is_static() && data_pshape[1].is_static() &&
        filters_pshape.rank().is_static() && filters_pshape[2].is_static())
    {
        auto n_data_channels = data_pshape[1].get_length();
        auto input_channels = filters_pshape[2].get_length();

        NGRAPH_CHECK((n_data_channels % input_channels) == 0);
        auto groups = n_data_channels / input_channels;
        group_dim = Dimension(groups);
    }
    return group_dim;
}

void op::v1::GroupConvolution::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_GroupConvolution_validate_and_infer_types);
    PartialShape data_batch_pshape = get_input_partial_shape(0);
    PartialShape filters_pshape = get_input_partial_shape(1);
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
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    NODE_VALIDATION_CHECK(
        this,
        (data_batch_pshape.rank().compatible(5) && filters_pshape.rank().compatible(6)) ||
            (data_batch_pshape.rank().compatible(4) && filters_pshape.rank().compatible(5)) ||
            (data_batch_pshape.rank().compatible(3) && filters_pshape.rank().compatible(4)),
        "Shapes for data batch and filters do not match. (data batch shape: ",
        data_batch_pshape,
        ", filters shape: ",
        filters_pshape,
        ").");

    PartialShape result_shape{PartialShape::dynamic()};
    if (data_batch_pshape.rank().is_static() || filters_pshape.rank().is_static())
    {
        const bool is_data_batch_ps_static = data_batch_pshape.rank().is_static();
        const auto output_ps_rank = is_data_batch_ps_static
                                        ? data_batch_pshape.rank().get_length()
                                        : filters_pshape.rank().get_length() - 1;
        const size_t num_spatial_dims = output_ps_rank - 2;

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

        NODE_VALIDATION_CHECK(this,
                              m_strides.size() == num_spatial_dims,
                              "Strides should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_dilations.size() == num_spatial_dims,
                              "Dilations should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_pads_begin.size() == num_spatial_dims &&
                                  m_pads_end.size() == num_spatial_dims,
                              "Pads should be defined for all and only spatial features.");

        if (data_batch_pshape.rank().is_static() && filters_pshape.rank().is_static())
        {
            auto data_in_channels_dim = data_batch_pshape[1];
            if (data_in_channels_dim.is_static())
            {
                auto groups_dim = filters_pshape[0];
                if (groups_dim.is_static() && filters_pshape[2].is_static())
                {
                    NODE_VALIDATION_CHECK(
                        this,
                        data_in_channels_dim.get_length() / groups_dim.get_length() ==
                            filters_pshape[2].get_length(),
                        "Input channels dimension of data batch has incompatible value "
                        "with filter shape.");
                }
                else if (groups_dim.is_static())
                {
                    NODE_VALIDATION_CHECK(
                        this,
                        data_in_channels_dim.get_length() % groups_dim.get_length() == 0,
                        "Input channels dimension of data batch not a multiple of group size.");
                }
            }
        }

        result_shape = std::vector<Dimension>(output_ps_rank, Dimension::dynamic());
        if (data_batch_pshape.rank().is_static())
        {
            result_shape[0] = data_batch_pshape[0]; // batch size
        }
        if (filters_pshape.rank().is_static() && filters_pshape.rank().get_length() > 2)
        {
            result_shape[1] = filters_pshape[0] * filters_pshape[1];
        }
        if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
        {
            bool auto_padding_applied = false;
            if (filters_pshape.rank().is_static() && filters_pshape.rank().get_length() > 2)
            {
                m_pads_begin.clear();
                m_pads_end.clear();

                const PartialShape filter_spatial_shape = [filters_pshape]() {
                    vector<Dimension> filter_dims{filters_pshape};
                    filter_dims.erase(filter_dims.begin(),
                                      filter_dims.begin() + 3); // Remove {GROUP, C_OUT, C_IN}
                    return PartialShape{filter_dims};
                }();

                if (filter_spatial_shape.is_static())
                {
                    auto_padding_applied = try_apply_auto_padding(data_batch_pshape,
                                                                  filter_spatial_shape.to_shape(),
                                                                  m_strides,
                                                                  m_dilations,
                                                                  m_auto_pad,
                                                                  m_pads_end,
                                                                  m_pads_begin);
                }
            }
            if (!auto_padding_applied)
            {
                set_output_type(0, result_et, result_shape);
                return;
            }
        }

        // we need to adjust channels input dim to reuse helpers for regular convolution
        PartialShape data_batch_ps = [&]() {
            auto shape = PartialShape{data_batch_pshape};
            auto groups = filters_pshape.rank().is_static() ? filters_pshape[0] : Dimension();
            if (groups.is_dynamic())
            {
                groups = infer_group_from_input_shapes(data_batch_pshape, filters_pshape);
            }
            if (data_batch_pshape.rank().is_static() && data_batch_pshape.rank().get_length())
            {
                if (data_batch_pshape[1].is_static() && groups.is_static())
                {
                    shape[1] = Dimension(data_batch_pshape[1].get_length() / groups.get_length());
                }
                else
                {
                    shape[1] = Dimension();
                }
            }
            return shape;
        }();

        // we need to adjust filters shape to reuse helpers for regular convolution
        PartialShape filters_ps = [&]() {
            auto shape = PartialShape{filters_pshape};
            if (shape.rank().is_static() && shape.rank().get_length() > 2)
            {
                auto groups = filters_pshape.rank().is_static() ? filters_pshape[0] : Dimension();
                if (groups.is_dynamic())
                {
                    groups = infer_group_from_input_shapes(data_batch_pshape, filters_pshape);
                }
                shape[1] = groups * shape[1];
                vector<Dimension> dim_vec{shape};
                dim_vec.erase(dim_vec.begin());
                shape = PartialShape{dim_vec};
            }
            return shape;
        }();

        result_shape =
            infer_convolution_forward(this,
                                      data_batch_ps,
                                      Strides(m_strides.size(), 1), // dummy data dilations
                                      m_pads_begin,
                                      m_pads_end,
                                      filters_ps,
                                      m_strides,
                                      m_dilations);
    }
    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::v1::GroupConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_GroupConvolution_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::GroupConvolution>(new_args.at(0),
                                             new_args.at(1),
                                             m_strides,
                                             m_pads_begin,
                                             m_pads_end,
                                             m_dilations,
                                             m_auto_pad);
}

//------------------------------------------------------------------------------
//                        v1::GroupConvolutionBackpropData
//------------------------------------------------------------------------------

NGRAPH_RTTI_DEFINITION(op::v1::GroupConvolutionBackpropData, "GroupConvolutionBackpropData", 1);

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData()
    : Op()
    , m_strides()
    , m_dilations()
    , m_pads_begin()
    , m_pads_end()
    , m_auto_pad()
    , m_output_padding()
{
}

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData(
    const Output<Node>& data,
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

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData(
    const Output<Node>& data,
    const Output<Node>& filters,
    const Output<Node>& output_shape,
    const Strides& strides,
    const Strides& dilations,
    const PadType& auto_pad,
    const CoordinateDiff& output_padding)
    : GroupConvolutionBackpropData(data,
                                   filters,
                                   output_shape,
                                   strides,
                                   CoordinateDiff(),
                                   CoordinateDiff(),
                                   dilations,
                                   auto_pad,
                                   output_padding)
{
}

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData(
    const Output<Node>& data,
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

bool ngraph::op::v1::GroupConvolutionBackpropData::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_GroupConvolutionBackpropData_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("output_padding", m_output_padding);
    return true;
}

bool op::v1::GroupConvolutionBackpropData::is_dynamic() const
{
    bool is_dynamic = Node::is_dynamic();
    if (inputs().size() == 3 && !is_dynamic)
    {
        return !has_and_set_equal_bounds(input_value(2));
    }
    return is_dynamic;
}

static Dimension infer_backprop_group_from_input_shapes(const PartialShape& data_pshape,
                                                        const PartialShape& filters_pshape)
{
    Dimension group_dim = Dimension();
    if (data_pshape.rank().is_static() && data_pshape[1].is_static() &&
        filters_pshape.rank().is_static() && filters_pshape[1].is_static())
    {
        auto n_data_channels = data_pshape[1].get_length();
        auto input_channels = filters_pshape[1].get_length();

        NGRAPH_CHECK((n_data_channels % input_channels) == 0);
        auto groups = n_data_channels / input_channels;
        group_dim = Dimension(groups);
    }
    return group_dim;
}

const PartialShape op::v1::GroupConvolutionBackpropData::get_convolution_output_shape() const
{
    auto data_pshape = get_input_partial_shape(0);
    auto filter_pshape = get_input_partial_shape(1);

    PartialShape shape;
    if (inputs().size() == 3)
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
        shape = PartialShape{vector<Dimension>(filter_pshape.rank().get_length() - 3)};
    }
    else
    {
        shape = PartialShape::dynamic();
    }
    return shape;
}

void op::v1::GroupConvolutionBackpropData::set_output_shape(const Shape& shape)
{
    this->input(2).replace_source_output(
        op::Constant::create(this->get_input_element_type(2), Shape{shape.size()}, shape)
            ->output(0));
}

void op::v1::GroupConvolutionBackpropData::infer_conv_backprop_output_spatial_shape(
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
    NGRAPH_CHECK(filters_shape.size() == num_spatial_dims && strides.size() == num_spatial_dims &&
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

void op::v1::GroupConvolutionBackpropData::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_GroupConvolutionBackpropData_validate_and_infer_types);
    const PartialShape& data_pshape = get_input_partial_shape(0);
    element::Type data_et = get_input_element_type(0);
    const PartialShape& filters_pshape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_et,
        ", filters element type: ",
        filters_et,
        ").");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    NODE_VALIDATION_CHECK(
        this,
        (data_pshape.rank().compatible(5) && filters_pshape.rank().compatible(6)) ||
            (data_pshape.rank().compatible(4) && filters_pshape.rank().compatible(5)) ||
            (data_pshape.rank().compatible(3) && filters_pshape.rank().compatible(4)),
        "Shapes for data batch and filters do not match. (data batch shape: ",
        data_pshape,
        ", filters shape: ",
        filters_pshape,
        ").");

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
    PartialShape output_spatial_pshape = get_convolution_output_shape();

    if (data_pshape.rank().is_static() || filters_pshape.rank().is_static())
    {
        const bool is_data_ps_static = data_pshape.rank().is_static();
        const auto output_ps_rank = is_data_ps_static ? data_pshape.rank().get_length()
                                                      : filters_pshape.rank().get_length() - 1;
        const size_t num_spatial_dims = output_ps_rank - 2;

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
                              m_strides.size() == num_spatial_dims,
                              "Strides should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_dilations.size() == num_spatial_dims,
                              "Dilations should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_pads_begin.size() == num_spatial_dims &&
                                  m_pads_end.size() == num_spatial_dims,
                              "Pads should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_output_padding.size() == num_spatial_dims,
                              "Output padding should be defined for all and only "
                              "spatial features.");

        if (data_pshape.rank().is_static() && filters_pshape.rank().is_static())
        {
            if (filters_pshape[0].is_static() && filters_pshape[1].is_static() &&
                data_pshape[1].is_static())
            {
                auto groups = filters_pshape[0].get_length();
                auto input_channels = filters_pshape[1].get_length();
                auto n_data_channels = data_pshape[1].get_length();

                NODE_VALIDATION_CHECK(this,
                                      n_data_channels % groups == 0,
                                      "Number of data channels not a multiple of group size.");
                NODE_VALIDATION_CHECK(this,
                                      n_data_channels / groups == input_channels,
                                      "Data second dimension has incompatible value "
                                      "with number of input channels.");
            }
        }

        if (is_output_shape_present && output_spatial_pshape.is_static())
        {
            Shape output_shape = output_spatial_pshape.to_shape();
            NODE_VALIDATION_CHECK(this,
                                  output_shape.size() == num_spatial_dims,
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
                                       filters_dims.begin() + 3); // remove {GROUPS, C_OUT, C_IN}
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
            // GROUPS * C_OUT
            auto n_out_channels = Dimension::dynamic();
            if (filters_pshape.rank().is_static())
            {
                auto group_dim = filters_pshape[0];
                if (!group_dim.is_static())
                {
                    group_dim = infer_backprop_group_from_input_shapes(data_pshape, filters_pshape);
                }
                n_out_channels = group_dim * filters_pshape[2];
            }
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
                return vector<Dimension>{std::next(data_dims.begin(), 2), std::end(data_dims)};
            }();

            auto filters_spatial_shape = [filters_pshape]() {
                vector<Dimension> filters_dims{filters_pshape};
                return vector<Dimension>{std::next(filters_dims.begin(), 3),
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
            // GROUPS * C_OUT
            auto n_out_channels = Dimension::dynamic();
            if (filters_pshape.rank().is_static())
            {
                auto group_dim = filters_pshape[0];
                if (!group_dim.is_static())
                {
                    group_dim = infer_backprop_group_from_input_shapes(data_pshape, filters_pshape);
                }
                n_out_channels = group_dim * filters_pshape[2];
            }
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
    op::v1::GroupConvolutionBackpropData::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_GroupConvolutionBackpropData_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 3)
    {
        return make_shared<v1::GroupConvolutionBackpropData>(new_args.at(0),
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
        return make_shared<v1::GroupConvolutionBackpropData>(new_args.at(0),
                                                             new_args.at(1),
                                                             m_strides,
                                                             m_pads_begin,
                                                             m_pads_end,
                                                             m_dilations,
                                                             m_auto_pad,
                                                             m_output_padding);
    }
}
