// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/deconvolution.hpp"
#include <memory>
#include "openvino/core/type/element_type.hpp"
#include "convolution_backprop_shape_inference.hpp"
#include "group_convolution_backprop_shape_inference.hpp"
#include "openvino/op/group_conv.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

Deconvolution::Deconvolution(const ov::Output<Node>& data_batch,
                         const ov::Output<Node>& filters,
                         const ov::Output<Node>& bias,
                         const ov::Strides& strides,
                         const ov::CoordinateDiff& pads_begin,
                         const ov::CoordinateDiff& pads_end,
                         const ov::Strides& dilations,
                         const int64_t& groups,
                         const ov::op::PadType& auto_pad,
                         const ov::element::Type& output_type,
                         const ov::CoordinateDiff& output_padding)
    : ov::op::util::ConvolutionBackPropBase({data_batch, filters, bias}, strides, pads_begin, pads_end, dilations, auto_pad, output_padding)
    , m_groups(groups)
    , m_asymmetric(false)
    , m_output_type(output_type) {
    validate_and_infer_types();
}

Deconvolution::Deconvolution(const ov::Output<Node>& data_batch,
                         const ov::Output<Node>& filters,
                         const ov::Output<Node>& bias,
                         const ov::Output<Node>& output_shape,
                         const ov::Strides& strides,
                         const ov::CoordinateDiff& pads_begin,
                         const ov::CoordinateDiff& pads_end,
                         const ov::Strides& dilations,
                         const int64_t& groups,
                         const ov::op::PadType& auto_pad,
                         const ov::element::Type& output_type,
                         const ov::CoordinateDiff& output_padding)
    : ov::op::util::ConvolutionBackPropBase({data_batch, filters, output_shape, bias}, strides, pads_begin, pads_end, dilations, auto_pad, output_padding)
    , m_groups(groups)
    , m_asymmetric(false)
    , m_output_type(output_type) {
    validate_and_infer_types();
}

bool Deconvolution::visit_attributes(ov::AttributeVisitor& visitor) {
    ov::op::util::ConvolutionBackPropBase::visit_attributes(visitor);
    visitor.on_attribute("groups", m_groups);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("asymmetric", m_asymmetric);
    return true;
}

void Deconvolution::validate_and_infer_types() {
    const auto& data_batch_et = get_input_element_type(0);
    const auto& filters_et = get_input_element_type(1);

    element::Type result_et;

    if (m_output_type != ov::element::undefined) {
        result_et = m_output_type;
    } else if (data_batch_et.compatible(filters_et)) {
        ov::element::Type::merge(result_et, data_batch_et, filters_et);
    } else if (data_batch_et == ov::element::u8 || data_batch_et == ov::element::i8) {
        result_et = ov::element::f32;
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    /* Remove last bias parameter from input shapes so shape infer does not think third bias argument is spatial dimension */
    auto input_shapes_copy = input_shapes;
    input_shapes_copy.pop_back();
    const auto output_shapes = intel_gpu::op::shape_infer(this, input_shapes_copy, m_pads_begin, m_pads_end);

    auto num_spatial = ov::op::convolution::calculate_num_spatial(this, input_shapes, output_shapes[0]);
    if (num_spatial != ov::op::util::num_spatial_undefined) {
        resize_attributes(num_spatial);
    }
    set_output_type(0, result_et, output_shapes[0]);
    set_num_spatial(num_spatial, input_shapes);
}

std::shared_ptr<Node> Deconvolution::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<Deconvolution>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             new_args.at(3),
                                             m_strides,
                                             m_pads_begin,
                                             m_pads_end,
                                             m_dilations,
                                             m_groups,
                                             m_auto_pad,
                                             m_output_type,
                                             m_output_padding);
    } else {
        return std::make_shared<Deconvolution>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             m_strides,
                                             m_pads_begin,
                                             m_pads_end,
                                             m_dilations,
                                             m_groups,
                                             m_auto_pad,
                                             m_output_type,
                                             m_output_padding);
    }
}

bool Deconvolution::has_groups() const {
    return m_groups > 0;
}

int64_t Deconvolution::get_groups() const {
    return m_groups;
}

bool Deconvolution::is_asymmetric() const {
    return m_asymmetric;
}

std::vector<ov::PartialShape> shape_infer(const Deconvolution* op,
                                          const std::vector<ov::PartialShape>& input_shapes,
                                          CoordinateDiff& pads_begin,
                                          CoordinateDiff& pads_end) {
   if (op->get_groups() > 0) {
        ov::op::v1::GroupConvolutionBackpropData tmp_op;
        tmp_op.set_strides(op->get_strides());
        tmp_op.set_dilations(op->get_dilations());
        tmp_op.set_auto_pad(op->get_auto_pad());
        tmp_op.set_output_padding(op->get_output_padding());

        return shape_infer(&tmp_op, input_shapes, pads_begin, pads_end);
   } else {
        ov::op::v1::ConvolutionBackpropData tmp_op;
        tmp_op.set_strides(op->get_strides());
        tmp_op.set_dilations(op->get_dilations());
        tmp_op.set_auto_pad(op->get_auto_pad());
        tmp_op.set_output_padding(op->get_output_padding());

        return shape_infer(&tmp_op, input_shapes, pads_begin, pads_end);
   }
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
