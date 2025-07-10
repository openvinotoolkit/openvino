// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/convolution.hpp"
#include <memory>
#include "openvino/core/type/element_type.hpp"
#include "convolution_shape_inference.hpp"
#include "group_convolution_shape_inference.hpp"
#include "openvino/op/group_conv.hpp"

namespace ov::intel_gpu::op {

Convolution::Convolution(const ov::Output<Node>& data_batch,
                         const ov::Output<Node>& filters,
                         const ov::Output<Node>& bias,
                         const ov::Strides& strides,
                         const ov::CoordinateDiff& pads_begin,
                         const ov::CoordinateDiff& pads_end,
                         const ov::Strides& dilations,
                         const int64_t& groups,
                         const ov::op::PadType& auto_pad,
                         const ov::element::Type& output_type)
    : ov::op::util::ConvolutionFwdPropBase({data_batch, filters, bias}, strides, pads_begin, pads_end, dilations, auto_pad)
    , m_groups(groups)
    , m_asymmetric(false)
    , m_output_type(output_type) {
    validate_and_infer_types();
}

Convolution::Convolution(const ov::Output<Node>& data_batch,
                         const ov::Output<Node>& filters,
                         const ov::Output<Node>& bias,
                         const ov::Output<Node>& activations_zero_point,
                         const ov::Output<Node>& weights_zero_point,
                         const ov::Output<Node>& compensations,
                         const ov::Strides& strides,
                         const ov::CoordinateDiff& pads_begin,
                         const ov::CoordinateDiff& pads_end,
                         const ov::Strides& dilations,
                         const int64_t& groups,
                         const ov::op::PadType& auto_pad,
                         const ov::element::Type& output_type)
    : ov::op::util::ConvolutionFwdPropBase({data_batch, filters, bias}, strides, pads_begin, pads_end, dilations, auto_pad)
    , m_groups(groups)
    , m_asymmetric(true)
    , m_output_type(output_type) {
    set_argument(Args::AZP, activations_zero_point);
    set_argument(Args::WZP, weights_zero_point);
    set_argument(Args::COMPENSATION, compensations);
    validate_and_infer_types();
}

bool Convolution::visit_attributes(ov::AttributeVisitor& visitor) {
    ov::op::util::ConvolutionFwdPropBase::visit_attributes(visitor);
    visitor.on_attribute("groups", m_groups);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("asymmetric", m_asymmetric);
    return true;
}

void Convolution::validate_and_infer_types() {
    const auto& data_batch_et = get_input_element_type(0);
    const auto& filters_et = get_input_element_type(1);

    element::Type result_et;
    if (m_output_type != ov::element::dynamic) {
        result_et = m_output_type;
    } else if (data_batch_et.compatible(filters_et)) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(result_et, data_batch_et, filters_et),
                              "Element types for data batch and filters do not match (data batch element type: ",
                              data_batch_et,
                              ", filters element type: ",
                              filters_et,
                              ").");
    } else if (data_batch_et == ov::element::u8 || data_batch_et == ov::element::i8) {
        result_et = ov::element::f32;
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    auto num_spatial = ov::op::convolution::calculate_num_spatial(this, input_shapes);
    if (num_spatial != ov::op::util::num_spatial_undefined) {
        resize_attributes(num_spatial);
    }

    const auto output_shapes = intel_gpu::op::shape_infer(this, input_shapes, m_pads_begin, m_pads_end);
    set_output_type(0, result_et, output_shapes[0]);
    set_num_spatial(num_spatial, input_shapes);
}

std::shared_ptr<Node> Convolution::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<Convolution>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             m_strides,
                                             m_pads_begin,
                                             m_pads_end,
                                             m_dilations,
                                             m_groups,
                                             m_auto_pad,
                                             m_output_type);
    } else {
        return std::make_shared<Convolution>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             new_args.at(3),
                                             new_args.at(4),
                                             new_args.at(5),
                                             m_strides,
                                             m_pads_begin,
                                             m_pads_end,
                                             m_dilations,
                                             m_groups,
                                             m_auto_pad,
                                             m_output_type);
    }
}

bool Convolution::has_groups() const {
    return m_groups > 0;
}

int64_t Convolution::get_groups() const {
    return m_groups;
}

bool Convolution::is_asymmetric() const {
    return m_asymmetric;
}

std::vector<ov::PartialShape> shape_infer(const Convolution* op,
                                          const std::vector<ov::PartialShape>& input_shapes,
                                          CoordinateDiff& pads_begin,
                                          CoordinateDiff& pads_end) {
   if (op->get_groups() > 0) {
        ov::op::v1::GroupConvolution tmp_op;
        tmp_op.set_strides(op->get_strides());
        tmp_op.set_dilations(op->get_dilations());
        tmp_op.set_auto_pad(op->get_auto_pad());

        return shape_infer(&tmp_op, input_shapes, pads_begin, pads_end);
   } else {
        ov::op::v1::Convolution tmp_op;
        tmp_op.set_strides(op->get_strides());
        tmp_op.set_dilations(op->get_dilations());
        tmp_op.set_auto_pad(op->get_auto_pad());

        return shape_infer(&tmp_op, input_shapes, pads_begin, pads_end);
   }
}

}  // namespace ov::intel_gpu::op
