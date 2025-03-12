// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/convolution.hpp"

#include "convolution_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;

namespace ov {
op::internal::Convolution::Convolution(const Output<Node>& data_batch,
                                       const Output<Node>& filters,
                                       const Strides& strides,
                                       const CoordinateDiff& pads_begin,
                                       const CoordinateDiff& pads_end,
                                       const Strides& dilations,
                                       const PadType& auto_pad)
    : op::util::ConvolutionFwdPropBase({data_batch, filters}, strides, pads_begin, pads_end, dilations, auto_pad) {
    constructor_validate_and_infer_types();
}

op::internal::Convolution::Convolution(const Output<Node>& data_batch,
                                       const Output<Node>& filters,
                                       const Output<Node>& bias,
                                       const Strides& strides,
                                       const CoordinateDiff& pads_begin,
                                       const CoordinateDiff& pads_end,
                                       const Strides& dilations,
                                       const PadType& auto_pad)
    : op::util::ConvolutionFwdPropBase({data_batch, filters, bias},
                                       strides,
                                       pads_begin,
                                       pads_end,
                                       dilations,
                                       auto_pad) {
    constructor_validate_and_infer_types();
}

bool op::internal::Convolution::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::internal::Convolution::validate_and_infer_types() {
    const auto& data_batch_et = get_input_element_type(0);
    const auto& filters_et = get_input_element_type(1);

    element::Type result_et;
    if (inputs().size() == 3) {
        const auto& bias_et = get_input_element_type(2);
        result_et = bias_et;
    } else {
        element::Type::merge(result_et, data_batch_et, filters_et);
    }

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element types must be numeric. Got: ",
                          result_et);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    auto num_spatial = convolution::calculate_num_spatial(this, input_shapes);
    if (num_spatial != util::num_spatial_undefined) {
        resize_attributes(num_spatial);
    }

    const auto output_shapes = op::shape_infer(this, input_shapes, m_pads_begin, m_pads_end);
    set_output_type(0, result_et, output_shapes[0]);
    set_num_spatial(num_spatial, input_shapes);
}

shared_ptr<Node> op::internal::Convolution::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return make_shared<internal::Convolution>(new_args.at(0),
                                                  new_args.at(1),
                                                  m_strides,
                                                  m_pads_begin,
                                                  m_pads_end,
                                                  m_dilations,
                                                  m_auto_pad);
    }
    return make_shared<internal::Convolution>(new_args.at(0),
                                              new_args.at(1),
                                              new_args.at(2),
                                              m_strides,
                                              m_pads_begin,
                                              m_pads_end,
                                              m_dilations,
                                              m_auto_pad);
}

}  // namespace ov
