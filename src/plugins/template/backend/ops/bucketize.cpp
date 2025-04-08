// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/bucketize.hpp"

#include "evaluate_node.hpp"

namespace bucketize_v3 {
template <ov::element::Type_t t1, ov::element::Type_t t2, ov::element::Type_t t3>
inline void evaluate(const std::shared_ptr<ov::op::v3::Bucketize>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    using T3 = typename ov::element_type_traits<t3>::value_type;

    ov::reference::bucketize<T1, T2, T3>(inputs[0].data<T1>(),
                                         inputs[1].data<T2>(),
                                         outputs[0].data<T3>(),
                                         op->get_input_shape(0),
                                         op->get_input_shape(1),
                                         op->get_with_right_bound());
}

static inline constexpr uint16_t getElementMask(ov::element::Type_t type1, ov::element::Type_t type2) {
    return (static_cast<uint8_t>(type1)) | (static_cast<uint8_t>(type2) << 8);
}
}  // namespace bucketize_v3

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v3::Bucketize>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (bucketize_v3::getElementMask(op->get_input_element_type(0), op->get_input_element_type(1))) {
    case bucketize_v3::getElementMask(ov::element::f32, ov::element::f32):
        bucketize_v3::evaluate<ov::element::f32, ov::element::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f32, ov::element::f16):
        bucketize_v3::evaluate<ov::element::f32, ov::element::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f32, ov::element::i32):
        bucketize_v3::evaluate<ov::element::f32, ov::element::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f32, ov::element::i64):
        bucketize_v3::evaluate<ov::element::f32, ov::element::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f32, ov::element::i8):
        bucketize_v3::evaluate<ov::element::f32, ov::element::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f32, ov::element::u8):
        bucketize_v3::evaluate<ov::element::f32, ov::element::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f16, ov::element::f32):
        bucketize_v3::evaluate<ov::element::f16, ov::element::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f16, ov::element::f16):
        bucketize_v3::evaluate<ov::element::f16, ov::element::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f16, ov::element::i32):
        bucketize_v3::evaluate<ov::element::f16, ov::element::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f16, ov::element::i64):
        bucketize_v3::evaluate<ov::element::f16, ov::element::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f16, ov::element::i8):
        bucketize_v3::evaluate<ov::element::f16, ov::element::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::f16, ov::element::u8):
        bucketize_v3::evaluate<ov::element::f16, ov::element::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i32, ov::element::f32):
        bucketize_v3::evaluate<ov::element::i32, ov::element::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i32, ov::element::f16):
        bucketize_v3::evaluate<ov::element::i32, ov::element::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i32, ov::element::i32):
        bucketize_v3::evaluate<ov::element::i32, ov::element::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i32, ov::element::i64):
        bucketize_v3::evaluate<ov::element::i32, ov::element::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i32, ov::element::i8):
        bucketize_v3::evaluate<ov::element::i32, ov::element::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i32, ov::element::u8):
        bucketize_v3::evaluate<ov::element::i32, ov::element::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i64, ov::element::f32):
        bucketize_v3::evaluate<ov::element::i64, ov::element::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i64, ov::element::f16):
        bucketize_v3::evaluate<ov::element::i64, ov::element::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i64, ov::element::i32):
        bucketize_v3::evaluate<ov::element::i64, ov::element::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i64, ov::element::i64):
        bucketize_v3::evaluate<ov::element::i64, ov::element::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i64, ov::element::i8):
        bucketize_v3::evaluate<ov::element::i64, ov::element::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i64, ov::element::u8):
        bucketize_v3::evaluate<ov::element::i64, ov::element::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i8, ov::element::f32):
        bucketize_v3::evaluate<ov::element::i8, ov::element::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i8, ov::element::f16):
        bucketize_v3::evaluate<ov::element::i8, ov::element::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i8, ov::element::i32):
        bucketize_v3::evaluate<ov::element::i8, ov::element::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i8, ov::element::i64):
        bucketize_v3::evaluate<ov::element::i8, ov::element::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i8, ov::element::i8):
        bucketize_v3::evaluate<ov::element::i8, ov::element::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::i8, ov::element::u8):
        bucketize_v3::evaluate<ov::element::i8, ov::element::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::u8, ov::element::f32):
        bucketize_v3::evaluate<ov::element::u8, ov::element::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::u8, ov::element::f16):
        bucketize_v3::evaluate<ov::element::u8, ov::element::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::u8, ov::element::i32):
        bucketize_v3::evaluate<ov::element::u8, ov::element::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::u8, ov::element::i64):
        bucketize_v3::evaluate<ov::element::u8, ov::element::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::u8, ov::element::i8):
        bucketize_v3::evaluate<ov::element::u8, ov::element::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ov::element::u8, ov::element::u8):
        bucketize_v3::evaluate<ov::element::u8, ov::element::u8, ET>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v3::Bucketize>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v3::Bucketize>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
