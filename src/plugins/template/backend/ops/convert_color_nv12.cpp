// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert_color_nv12.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v8::NV12toRGB>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    return ov::reference::color_convert_nv12<ET>(op,
                                                 outputs,
                                                 inputs,
                                                 ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_RGB);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v8::NV12toBGR>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    return ov::reference::color_convert_nv12<ET>(op,
                                                 outputs,
                                                 inputs,
                                                 ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_BGR);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v8::I420toRGB>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    return ov::reference::color_convert_i420<ET>(op,
                                                 outputs,
                                                 inputs,
                                                 ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_RGB);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v8::I420toBGR>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    return ov::reference::color_convert_i420<ET>(op,
                                                 outputs,
                                                 inputs,
                                                 ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_BGR);
}

template <>
bool evaluate_node<ov::op::v8::NV12toRGB>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v8::NV12toRGB>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

template <>
bool evaluate_node<ov::op::v8::NV12toBGR>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v8::NV12toBGR>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

template <>
bool evaluate_node<ov::op::v8::I420toRGB>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v8::I420toRGB>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

template <>
bool evaluate_node<ov::op::v8::I420toBGR>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v8::I420toBGR>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
