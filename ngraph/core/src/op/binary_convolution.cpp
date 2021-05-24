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
                          data_batch_et.is_real() || data_batch_et.is_integral_number(),
                          "Data batch element type must be numeric. Got: ",
                          data_batch_et);

    // TODO: Add NodeValidationCheck to filters et once u1 is supported in nGraph Python API
    // (#52715)

    Rank result_ps_rank;
    NODE_VALIDATION_CHECK(
        this,
        Rank::merge(result_ps_rank, data_batch_pshape.rank(), filters_pshape.rank()),
        "Data batch and filters inputs must have same rank. Got: ",
        data_batch_pshape,
        " and ",
        filters_pshape);

    PartialShape result_shape =
        validate_and_infer_convolution_forward_output_shape(this,
                                                            result_ps_rank,
                                                            data_batch_pshape,
                                                            filters_pshape,
                                                            m_auto_pad,
                                                            m_strides,
                                                            m_dilations,
                                                            m_pads_begin,
                                                            m_pads_end);
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
    NGRAPH_API EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>&
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
} // namespace ngraph

op::v1::BinaryConvolution::BinaryConvolutionMode
    op::v1::BinaryConvolution::mode_from_string(const std::string& mode) const
{
    return as_enum<BinaryConvolutionMode>(mode);
}
