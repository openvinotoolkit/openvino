// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"
#include "openvino/op/roi_align.hpp"
#include "openvino/reference/roi_align.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v15::ROIAlignRotated>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    std::vector<int64_t> batch_indices_vec_scaled_up = get_integers(inputs[2], inputs[2].get_shape());
    ov::reference::roi_align<T, ov::reference::roi_policy::ROIAlignRotatedOpDefPolicy>(
        inputs[0].data<const T>(),
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
        ov::op::v3::ROIAlign::PoolingMode::AVG,
        ov::op::v9::ROIAlign::AlignedMode::ASYMMETRIC,
        op->get_clockwise_mode());
    return true;
}

template <>
bool evaluate_node<ov::op::v15::ROIAlignRotated>(std::shared_ptr<ov::Node> node,
                                                 ov::TensorVector& outputs,
                                                 const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

#define CASE(type)          \
    case ov::element::type: \
        return evaluate<ov::element::type>(ov::as_type_ptr<ov::op::v15::ROIAlignRotated>(node), outputs, inputs);

    switch (element_type) {
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(f64);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
#undef CASE
}
