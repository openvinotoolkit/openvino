// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert_color_to_nv12.hpp"

#include "evaluate_node.hpp"
#include "openvino/op/bgr_to_nv12.hpp"
#include "openvino/op/rgb_to_nv12.hpp"

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v16::RGBtoNV12>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    return ov::reference::color_convert_to_nv12<ET>(op,
                                                    outputs,
                                                    inputs,
                                                    ov::op::util::ConvertColorToNV12Base::ColorConversion::RGB_TO_NV12);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v16::BGRtoNV12>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    return ov::reference::color_convert_to_nv12<ET>(op,
                                                    outputs,
                                                    inputs,
                                                    ov::op::util::ConvertColorToNV12Base::ColorConversion::BGR_TO_NV12);
}

template <>
bool evaluate_node<ov::op::v16::RGBtoNV12>(std::shared_ptr<ov::Node> node,
                                           ov::TensorVector& outputs,
                                           const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v16::RGBtoNV12>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}

template <>
bool evaluate_node<ov::op::v16::BGRtoNV12>(std::shared_ptr<ov::Node> node,
                                           ov::TensorVector& outputs,
                                           const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v16::BGRtoNV12>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
