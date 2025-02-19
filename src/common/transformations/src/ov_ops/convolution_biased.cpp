// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/convolution_biased.hpp"

#include "convolution_backprop_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;

namespace ov {
op::internal::ConvolutionBiased::ConvolutionBiased(const Output<Node>& data_batch,
                                                   const Output<Node>& filters,
                                                   const Output<Node>& bias,
                                                   const Strides& strides,
                                                   const CoordinateDiff& pads_begin,
                                                   const CoordinateDiff& pads_end,
                                                   const Strides& dilations,
                                                   const PadType& auto_pad)
    : op::util::ConvolutionFwdPropBase({data_batch, filters, bias}, strides, pads_begin, pads_end, dilations, auto_pad) {
    constructor_validate_and_infer_types();
}

bool op::internal::ConvolutionBiased::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::internal::ConvolutionBiased::validate_and_infer_types() {
    util::ConvolutionFwdPropBase::validate_and_infer_types();
}

shared_ptr<Node> op::internal::ConvolutionBiased::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<internal::ConvolutionBiased>(new_args.at(0),
                                                    new_args.at(1),
                                                    new_args.at(2),
                                                    m_strides,
                                                    m_pads_begin,
                                                    m_pads_end,
                                                    m_dilations,
                                                    m_auto_pad);
}

}  // namespace ov
