// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/binary_convolution.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::BinaryConvolution::type_info;

op::v1::BinaryConvolution::BinaryConvolution(const Output<Node>& data,
                                             const Output<Node>& kernel,
                                             const Strides& strides,
                                             const CoordinateDiff& pads_begin,
                                             const CoordinateDiff& pads_end,
                                             const Strides& dilations,
                                             BinaryConvolutionMode mode,
                                             float pad_value,
                                             const PadType& auto_pad)
    : Op({data, kernel})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_mode(mode)
    , m_pad_value(pad_value)
    , m_auto_pad(auto_pad)
{
    constructor_validate_and_infer_types();
}

op::v1::BinaryConvolution::BinaryConvolution(const Output<Node>& data,
                                             const Output<Node>& kernel,
                                             const Strides& strides,
                                             const CoordinateDiff& pads_begin,
                                             const CoordinateDiff& pads_end,
                                             const Strides& dilations,
                                             const std::string& mode,
                                             float pad_value,
                                             const PadType& auto_pad)
    : Op({data, kernel})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_mode(mode_from_string(mode))
    , m_pad_value(pad_value)
    , m_auto_pad(auto_pad)
{
    constructor_validate_and_infer_types();
}

void op::v1::BinaryConvolution::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_BinaryConvolution_validate_and_infer_types);
    const PartialShape& data_batch_pshape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& filters_pshape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          data_batch_et.is_real(),
                          "Data batch element type must be float point. Got: ",
                          data_batch_et);

    // TODO: Add NodeValidationCheck to filters et once u1 is supported in nGraph Python API
    // (#49517)

    NODE_VALIDATION_CHECK(this,
                          data_batch_pshape.rank().compatible(filters_pshape.rank()),
                          "Shapes for data batch and filters must have same rank. Got: ",
                          data_batch_pshape,
                          "and ",
                          filters_pshape);

    if (m_strides.size() == 0)
    {
        m_strides = conv_default_strides(this, data_batch_pshape, filters_pshape);
    }

    if (m_dilations.size() == 0)
    {
        m_dilations = conv_default_strides(this, data_batch_pshape, filters_pshape);
    }

    if (m_pads_begin.size() == 0)
    {
        m_pads_begin = conv_default_padding(this, data_batch_pshape, filters_pshape);
    }

    if (m_pads_end.size() == 0)
    {
        m_pads_end = conv_default_padding(this, data_batch_pshape, filters_pshape);
    }

    PartialShape result_shape = PartialShape::dynamic();
    if (data_batch_pshape.rank().is_static() || filters_pshape.rank().is_static())
    {
        const bool is_data_batch_ps_static = data_batch_pshape.rank().is_static();
        const auto output_ps_rank =
            is_data_batch_ps_static ? data_batch_pshape.rank() : filters_pshape.rank();
        const auto num_spatial_dims = output_ps_rank.get_length() - 2;

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

        result_shape = std::vector<Dimension>(output_ps_rank.get_length(), Dimension::dynamic());
        if (data_batch_pshape.rank().is_static())
        {
            result_shape[0] = data_batch_pshape[0]; // batch size
        }
        if (filters_pshape.rank().is_static())
        {
            result_shape[1] = filters_pshape[0]; // filter channel size
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
                    filter_dims.erase(filter_dims.begin(), filter_dims.begin() + 2); // Remove {O,I}
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
                set_output_type(0, data_batch_et, result_shape);
                return;
            }
        }

        result_shape = infer_convolution_forward(this,
                                                 data_batch_pshape,
                                                 Strides(num_spatial_dims, 1),
                                                 m_pads_begin,
                                                 m_pads_end,
                                                 filters_pshape,
                                                 m_strides,
                                                 m_dilations);
    }
    set_output_type(0, data_batch_et, result_shape);
}

shared_ptr<Node>
    op::v1::BinaryConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_BinaryConvolution_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::BinaryConvolution>(new_args.at(0),
                                              new_args.at(1),
                                              m_strides,
                                              m_pads_begin,
                                              m_pads_end,
                                              m_dilations,
                                              m_mode,
                                              m_pad_value,
                                              m_auto_pad);
}

bool op::v1::BinaryConvolution::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_BinaryConvolution_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("pad_value", m_pad_value);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

namespace ngraph
{
    template <>
    EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>&
        EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>::get()
    {
        static auto enum_names = EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>(
            "op::v1::BinaryConvolution::BinaryConvolutionMode",
            {{"xnor-popcount", op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<op::v1::BinaryConvolution::BinaryConvolutionMode>::type_info;

    std::ostream& operator<<(std::ostream& s,
                             const op::v1::BinaryConvolution::BinaryConvolutionMode& type)
    {
        return s << as_string(type);
    }
}

op::v1::BinaryConvolution::BinaryConvolutionMode
    op::v1::BinaryConvolution::mode_from_string(const std::string& mode) const
{
    return as_enum<BinaryConvolutionMode>(mode);
}
