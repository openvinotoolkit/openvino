// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/deformable_convolution.hpp"
#include "itt.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
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
                                                     const bool use_bilinear_interpolation_padding)
    : DeformableConvolutionBase({arg, offsets, filters},
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                group,
                                deformable_group)
    , m_use_bilinear_interpolation_padding(use_bilinear_interpolation_padding)
{
}

op::v8::DeformableConvolution::DeformableConvolution(const Output<Node>& arg,
                                                     const Output<Node>& offsets,
                                                     const Output<Node>& filters,
                                                     const Output<Node>& modulation_scalars,
                                                     const Strides& strides,
                                                     const CoordinateDiff& pads_begin,
                                                     const CoordinateDiff& pads_end,
                                                     const Strides& dilations,
                                                     const op::PadType& auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group,
                                                     const bool use_bilinear_interpolation_padding)
    : DeformableConvolutionBase({arg, offsets, filters, modulation_scalars},
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                group,
                                deformable_group)
    , m_use_bilinear_interpolation_padding(use_bilinear_interpolation_padding)
{
}

bool op::v8::DeformableConvolution::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("use_bilinear_interpolation_padding",
                         m_use_bilinear_interpolation_padding);
    return DeformableConvolutionBase::visit_attributes(visitor);
}

void op::v8::DeformableConvolution::validate_and_infer_types()
{
    DeformableConvolutionBase::validate_and_infer_types();
}

std::shared_ptr<Node>
    op::v8::DeformableConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(DeformableConvolutionBase_clone_with_new_inputs);
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
                                                       m_use_bilinear_interpolation_padding);
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
                                                       m_use_bilinear_interpolation_padding);
    }
}
/*
namespace deformable_convolution
{
 // evaluate method
}*/

/*bool op::v8::DeformableConvolution::evaluate(const HostTensorVector &outputs, const
HostTensorVector &inputs) const { NGRAPH_OP_SCOPE(v8_DeformableConvolution);

    return true;
}*/

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
}

std::shared_ptr<Node>
    op::v1::DeformableConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(DeformableConvolutionBase_clone_with_new_inputs);
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
