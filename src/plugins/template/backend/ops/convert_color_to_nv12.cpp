// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert_color_to_nv12.hpp"

#include "evaluate_node.hpp"
#include "openvino/op/bgr_to_nv12.hpp"
#include "openvino/op/rgb_to_nv12.hpp"
#include "rgb_bgr_to_nv12_shape_inference.hpp"

namespace {
template <ov::element::Type_t ET, bool IsRGB>
bool evaluate_nv12(const ov::op::util::ConvertColorToNV12Base* op,
                   ov::TensorVector& outputs,
                   const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shapes = ov::op::shape_infer(op, input_shapes);
    for (size_t i = 0; i < outputs.size(); ++i)
        outputs[i].set_shape(output_shapes[i].to_shape());

    const auto& rgb_tensor = inputs[0];
    const auto batch_size = rgb_tensor.get_shape()[0];
    const auto image_h = rgb_tensor.get_shape()[1];
    const auto image_w = rgb_tensor.get_shape()[2];
    const bool single_plane = op->get_output_size() == 1;

    ov::reference::color_convert_to_nv12<T, IsRGB>(rgb_tensor.data<T>(),
                                                   outputs[0].data<T>(),
                                                   single_plane ? nullptr : outputs[1].data<T>(),
                                                   batch_size,
                                                   image_h,
                                                   image_w,
                                                   single_plane);
    return true;
}
}  // namespace

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v17::RGBtoNV12>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    return evaluate_nv12<ET, true>(op.get(), outputs, inputs);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v17::BGRtoNV12>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    return evaluate_nv12<ET, false>(op.get(), outputs, inputs);
}

template <>
bool evaluate_node<ov::op::v17::RGBtoNV12>(std::shared_ptr<ov::Node> node,
                                           ov::TensorVector& outputs,
                                           const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v17::RGBtoNV12>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v17::RGBtoNV12>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v17::RGBtoNV12>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v17::RGBtoNV12>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v17::RGBtoNV12>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}

template <>
bool evaluate_node<ov::op::v17::BGRtoNV12>(std::shared_ptr<ov::Node> node,
                                           ov::TensorVector& outputs,
                                           const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v17::BGRtoNV12>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v17::BGRtoNV12>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v17::BGRtoNV12>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v17::BGRtoNV12>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v17::BGRtoNV12>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}
