// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/deformable_psroi_pooling.hpp"

#include "evaluate_node.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::DeformablePSROIPooling>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    NGRAPH_CHECK(inputs.size() > 1 && inputs[1]->get_shape().size() == 2,
                 "2D tensor must be provided as second input. ");
    outputs[0]->set_shape({inputs[1]->get_shape()[0],
                           static_cast<size_t>(op->get_output_dim()),
                           static_cast<size_t>(op->get_group_size()),
                           static_cast<size_t>(op->get_group_size())});

    const bool has_offset_intput = inputs.size() == 3;
    if (has_offset_intput) {
        ov::reference::deformable_psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                                   inputs[0]->get_shape(),
                                                   inputs[1]->get_data_ptr<T>(),
                                                   inputs[1]->get_shape(),
                                                   inputs[2]->get_data_ptr<T>(),
                                                   inputs[2]->get_shape(),
                                                   outputs[0]->get_data_ptr<T>(),
                                                   outputs[0]->get_shape(),
                                                   op->get_mode(),
                                                   op->get_spatial_scale(),
                                                   op->get_spatial_bins_x(),
                                                   op->get_spatial_bins_y(),
                                                   op->get_trans_std(),
                                                   op->get_part_size());
    } else {
        ov::reference::deformable_psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                                   inputs[0]->get_shape(),
                                                   inputs[1]->get_data_ptr<T>(),
                                                   inputs[1]->get_shape(),
                                                   nullptr,
                                                   ngraph::Shape(),
                                                   outputs[0]->get_data_ptr<T>(),
                                                   outputs[0]->get_shape(),
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
bool evaluate_node<ngraph::op::v1::DeformablePSROIPooling>(std::shared_ptr<ngraph::Node> node,
                                                           const ngraph::HostTensorVector& outputs,
                                                           const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
