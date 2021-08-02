// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/deformable_convolution.hpp"
#include "itt.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/runtime/reference/deformable_convolution.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::DeformableConvolution,
                       "DeformableConvolution",
                       1,
                       op::util::DeformableConvolutionBase);
NGRAPH_RTTI_DEFINITION(op::v8::DeformableConvolution,
                       "DeformableConvolution",
                       8,
                       op::util::DeformableConvolutionBase);

op::v8::DeformableConvolution::DeformableConvolution(const Output<Node>& arg,
                                                     const Output<Node>& offsets,
                                                     const Output<Node>& filters,
                                                     const Strides& strides,
                                                     const CoordinateDiff& pads_begin,
                                                     const CoordinateDiff& pads_end,
                                                     const Strides& dilations,
                                                     const op::PadType& auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group,
                                                     const bool bilinear_interpolation_pad)
    : DeformableConvolutionBase({arg, offsets, filters},
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                group,
                                deformable_group)
    , m_bilinear_interpolation_pad(bilinear_interpolation_pad)
{
    constructor_validate_and_infer_types();
}

op::v8::DeformableConvolution::DeformableConvolution(const Output<Node>& arg,
                                                     const Output<Node>& offsets,
                                                     const Output<Node>& filters,
                                                     const Output<Node>& mask,
                                                     const Strides& strides,
                                                     const CoordinateDiff& pads_begin,
                                                     const CoordinateDiff& pads_end,
                                                     const Strides& dilations,
                                                     const op::PadType& auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group,
                                                     const bool bilinear_interpolation_pad)
    : DeformableConvolutionBase({arg, offsets, filters, mask},
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                group,
                                deformable_group)
    , m_bilinear_interpolation_pad(bilinear_interpolation_pad)
{
    constructor_validate_and_infer_types();
}

bool op::v8::DeformableConvolution::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(DeformableConvolution_v8_visit_attributes);
    visitor.on_attribute("bilinear_interpolation_pad", m_bilinear_interpolation_pad);
    return DeformableConvolutionBase::visit_attributes(visitor);
}

void op::v8::DeformableConvolution::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(DeformableConvolution_v8_validate_and_infer_types);

    DeformableConvolutionBase::validate_and_infer_types();
    if (inputs().size() == 4)
    {
        const PartialShape& data_pshape = get_input_partial_shape(0);
        const PartialShape& filters_pshape = get_input_partial_shape(2);
        const PartialShape& mask_pshape = get_input_partial_shape(3);
        element::Type mask_et = get_input_element_type(3);

        NODE_VALIDATION_CHECK(this,
                              mask_et.is_real() || mask_et.is_integral_number(),
                              "Element type of Mask input must be numeric. Got: ",
                              mask_et);

        NODE_VALIDATION_CHECK(this,
                              mask_pshape.rank().compatible(4),
                              "Mask input must be of rank 4. Got: ",
                              mask_pshape.rank());

        if (mask_pshape.rank().is_static() && mask_pshape[1].is_static())
        {
            if (filters_pshape.rank().is_static() && filters_pshape[2].is_static() &&
                filters_pshape[3].is_static())
            {
                auto offsets_channels = m_deformable_group * filters_pshape[2].get_length() *
                                        filters_pshape[3].get_length();
                NODE_VALIDATION_CHECK(this,
                                      mask_pshape[1].get_length() == offsets_channels,
                                      "The channels dimension of mask input is not "
                                      "compatible with filters and 'deformable group' attribute. "
                                      "Mask input shape: ",
                                      mask_pshape,
                                      ", deformable 'group' attribute value: ",
                                      m_deformable_group,
                                      ", filters shape: ",
                                      filters_pshape);
            }
            // At least we can check if mask channels is evenly divisible by deformable
            // group attribute
            NODE_VALIDATION_CHECK(this,
                                  mask_pshape[1].get_length() % m_deformable_group == 0,
                                  "The channels dimension of mask input must be "
                                  "evenly divisible by the 'deformable group' value along the "
                                  "channels axis. Offsets input shape: ",
                                  mask_pshape,
                                  ", 'deformable group' attribute value: ",
                                  m_deformable_group);

            if (data_pshape.rank().is_static())
            {
                NODE_VALIDATION_CHECK(
                    this,
                    mask_pshape[0].compatible(data_pshape[0]),
                    "Data batch and mask batch dimension must be same value. Got: ",
                    mask_pshape[0],
                    " and ",
                    data_pshape[0]);
            }
        }

        PartialShape result_pshape = get_output_partial_shape(0);
        if (result_pshape.rank().is_static() && mask_pshape.rank().is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  result_pshape[2].compatible(mask_pshape[2]) &&
                                      result_pshape[3].compatible(mask_pshape[3]),
                                  "Spatial dimensions of mask and output must be equal. Got: ",
                                  mask_pshape[2],
                                  ", ",
                                  mask_pshape[3],
                                  " and ",
                                  result_pshape[2],
                                  ", ",
                                  result_pshape[3]);
        }
    }
}

std::shared_ptr<Node>
    op::v8::DeformableConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(DeformableConvolution_v8_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(
        this, new_args.size() >= 3 && new_args.size() <= 4, "Number of inputs must be 3 or 4");
    switch (new_args.size())
    {
    case 3:
        return std::make_shared<DeformableConvolution>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       m_strides,
                                                       m_pads_begin,
                                                       m_pads_end,
                                                       m_dilations,
                                                       m_auto_pad,
                                                       m_group,
                                                       m_deformable_group,
                                                       m_bilinear_interpolation_pad);
    default:
        return std::make_shared<DeformableConvolution>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       new_args.at(3),
                                                       m_strides,
                                                       m_pads_begin,
                                                       m_pads_end,
                                                       m_dilations,
                                                       m_auto_pad,
                                                       m_group,
                                                       m_deformable_group,
                                                       m_bilinear_interpolation_pad);
    }
}

namespace deformable_convolution
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorVector& inputs,
                         const HostTensorPtr& out,
                         const Strides& strides,
                         const CoordinateDiff& pads_begin,
                         const CoordinateDiff& pads_end,
                         const Strides& dilations,
                         const ngraph::op::PadType& auto_pad,
                         const int64_t group,
                         const int64_t deformable_group,
                         const bool use_bilinear_interpolation_padding)
    {
        using T = typename element_type_traits<ET>::value_type;
        if (inputs.size() == 3)
        {
            runtime::reference::deformable_convolution<T>(inputs[0]->get_data_ptr<ET>(),
                                                          inputs[1]->get_data_ptr<ET>(),
                                                          inputs[2]->get_data_ptr<ET>(),
                                                          out->get_data_ptr<ET>(),
                                                          inputs[0]->get_shape(),
                                                          inputs[1]->get_shape(),
                                                          inputs[2]->get_shape(),
                                                          out->get_shape(),
                                                          strides,
                                                          dilations,
                                                          pads_begin,
                                                          pads_end,
                                                          group,
                                                          deformable_group,
                                                          use_bilinear_interpolation_padding);
        }
        else if (inputs.size() == 4)
        {
            runtime::reference::deformable_convolution<T>(inputs[0]->get_data_ptr<ET>(),
                                                          inputs[1]->get_data_ptr<ET>(),
                                                          inputs[2]->get_data_ptr<ET>(),
                                                          inputs[3]->get_data_ptr<ET>(),
                                                          out->get_data_ptr<ET>(),
                                                          inputs[0]->get_shape(),
                                                          inputs[1]->get_shape(),
                                                          inputs[2]->get_shape(),
                                                          inputs[3]->get_shape(),
                                                          out->get_shape(),
                                                          strides,
                                                          dilations,
                                                          pads_begin,
                                                          pads_end,
                                                          group,
                                                          deformable_group,
                                                          use_bilinear_interpolation_padding);
        }

        return true;
    }

    bool evaluate_deformable_convolution(const HostTensorVector& inputs,
                                         const HostTensorPtr& out,
                                         const Strides& strides,
                                         const Strides& dilations,
                                         const CoordinateDiff& pads_begin,
                                         const CoordinateDiff& pads_end,
                                         const ngraph::op::PadType& auto_pad,
                                         const int64_t group,
                                         const int64_t deformable_group,
                                         const bool use_bilinear_interpolation_padding)
    {
        bool rc = true;
        switch (inputs[0]->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_deformable_convolution,
                             f32,
                             inputs,
                             out,
                             strides,
                             pads_begin,
                             pads_end,
                             dilations,
                             auto_pad,
                             group,
                             deformable_group,
                             use_bilinear_interpolation_padding);
            NGRAPH_TYPE_CASE(evaluate_deformable_convolution,
                             f16,
                             inputs,
                             out,
                             strides,
                             pads_begin,
                             pads_end,
                             dilations,
                             auto_pad,
                             group,
                             deformable_group,
                             use_bilinear_interpolation_padding);
            NGRAPH_TYPE_CASE(evaluate_deformable_convolution,
                             i32,
                             inputs,
                             out,
                             strides,
                             pads_begin,
                             pads_end,
                             dilations,
                             auto_pad,
                             group,
                             deformable_group,
                             use_bilinear_interpolation_padding);
            NGRAPH_TYPE_CASE(evaluate_deformable_convolution,
                             i16,
                             inputs,
                             out,
                             strides,
                             pads_begin,
                             pads_end,
                             dilations,
                             auto_pad,
                             group,
                             deformable_group,
                             use_bilinear_interpolation_padding);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace deformable_convolution

bool op::v8::DeformableConvolution::evaluate(const HostTensorVector& outputs,
                                             const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(DeformableConvolution_v8_evaluate);
    deformable_convolution::evaluate_deformable_convolution(inputs,
                                                            outputs[0],
                                                            get_strides(),
                                                            get_dilations(),
                                                            get_pads_begin(),
                                                            get_pads_end(),
                                                            get_auto_pad(),
                                                            get_group(),
                                                            get_deformable_group(),
                                                            get_bilinear_interpolation_pad());
    return true;
}

bool op::v8::DeformableConvolution::has_evaluate() const
{
    NGRAPH_OP_SCOPE(DeformableConvolution_v8_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::f16:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}

op::v1::DeformableConvolution::DeformableConvolution(const Output<Node>& arg,
                                                     const Output<Node>& offsets,
                                                     const Output<Node>& filters,
                                                     const Strides& strides,
                                                     const CoordinateDiff& pads_begin,
                                                     const CoordinateDiff& pads_end,
                                                     const Strides& dilations,
                                                     const op::PadType& auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group)
    : DeformableConvolutionBase({arg, offsets, filters},
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                group,
                                deformable_group)
{
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node>
    op::v1::DeformableConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(DeformableConvolution_v1_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<DeformableConvolution>(new_args.at(0),
                                                   new_args.at(1),
                                                   new_args.at(2),
                                                   m_strides,
                                                   m_pads_begin,
                                                   m_pads_end,
                                                   m_dilations,
                                                   m_auto_pad,
                                                   m_group,
                                                   m_deformable_group);
}
