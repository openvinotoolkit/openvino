// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/roi_align.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v9::ROIAlign>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    std::vector<int64_t> batch_indices_vec_scaled_up = get_integers(inputs[2], inputs[2].get_shape());
    ov::op::v3::ROIAlign::PoolingMode m_mode_v3;
    switch (op->get_mode()) {
    case ov::op::v9::ROIAlign::PoolingMode::AVG: {
        m_mode_v3 = ov::op::v3::ROIAlign::PoolingMode::AVG;
        break;
    }
    case ov::op::v9::ROIAlign::PoolingMode::MAX: {
        m_mode_v3 = ov::op::v3::ROIAlign::PoolingMode::MAX;
        break;
    }
    default: {
        OPENVINO_THROW("unsupported PoolingMode ");
    }
    }
    ov::reference::roi_align<T>(inputs[0].data<const T>(),
                                inputs[1].data<const T>(),
                                batch_indices_vec_scaled_up.data(),
                                outputs[0].data<T>(),
                                op->get_input_shape(0),
                                op->get_input_shape(1),
                                op->get_input_shape(2),
                                op->get_output_shape(0),
                                op->get_pooled_h(),
                                op->get_pooled_w(),
                                op->get_sampling_ratio(),
                                op->get_spatial_scale(),
                                m_mode_v3,
                                op->get_aligned_mode());
    return true;
}

template <>
bool evaluate_node<ov::op::v9::ROIAlign>(std::shared_ptr<ov::Node> node,
                                         ov::TensorVector& outputs,
                                         const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v9::ROIAlign>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
