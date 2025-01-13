// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/deformable_psroi_pooling.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::DeformablePSROIPooling>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    OPENVINO_ASSERT(inputs.size() > 1 && inputs[1].get_shape().size() == 2,
                    "2D tensor must be provided as second input. ");
    outputs[0].set_shape({inputs[1].get_shape()[0],
                          static_cast<size_t>(op->get_output_dim()),
                          static_cast<size_t>(op->get_group_size()),
                          static_cast<size_t>(op->get_group_size())});

    const bool has_offset_intput = inputs.size() == 3;
    if (has_offset_intput) {
        ov::reference::deformable_psroi_pooling<T>(inputs[0].data<T>(),
                                                   inputs[0].get_shape(),
                                                   inputs[1].data<T>(),
                                                   inputs[1].get_shape(),
                                                   inputs[2].data<T>(),
                                                   inputs[2].get_shape(),
                                                   outputs[0].data<T>(),
                                                   outputs[0].get_shape(),
                                                   op->get_mode(),
                                                   op->get_spatial_scale(),
                                                   op->get_spatial_bins_x(),
                                                   op->get_spatial_bins_y(),
                                                   op->get_trans_std(),
                                                   op->get_part_size());
    } else {
        ov::reference::deformable_psroi_pooling<T>(inputs[0].data<T>(),
                                                   inputs[0].get_shape(),
                                                   inputs[1].data<T>(),
                                                   inputs[1].get_shape(),
                                                   nullptr,
                                                   ov::Shape(),
                                                   outputs[0].data<T>(),
                                                   outputs[0].get_shape(),
                                                   op->get_mode(),
                                                   op->get_spatial_scale(),
                                                   op->get_spatial_bins_x(),
                                                   op->get_spatial_bins_y(),
                                                   op->get_trans_std(),
                                                   op->get_part_size());
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v1::DeformablePSROIPooling>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node),
                                              outputs,
                                              inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
