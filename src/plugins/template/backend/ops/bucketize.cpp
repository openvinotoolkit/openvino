// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/bucketize.hpp"

#include "evaluate_node.hpp"

namespace bucketize_v3 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2, ngraph::element::Type_t t3>
inline void evaluate(const std::shared_ptr<ngraph::op::v3::Bucketize>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
    using T3 = typename ngraph::element_type_traits<t3>::value_type;

    ngraph::runtime::reference::bucketize<T1, T2, T3>(inputs[0]->get_data_ptr<T1>(),
                                                      inputs[1]->get_data_ptr<T2>(),
                                                      outputs[0]->get_data_ptr<T3>(),
                                                      op->get_input_shape(0),
                                                      op->get_input_shape(1),
                                                      op->get_with_right_bound());
}

static inline constexpr uint16_t getElementMask(ngraph::element::Type_t type1, ngraph::element::Type_t type2) {
    return (static_cast<uint8_t>(type1)) | (static_cast<uint8_t>(type2) << 8);
}
}  // namespace bucketize_v3

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::Bucketize>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (bucketize_v3::getElementMask(op->get_input_element_type(0), op->get_input_element_type(1))) {
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ngraph::op::v3::Bucketize>(std::shared_ptr<ngraph::Node> node,
                                              const ngraph::HostTensorVector& outputs,
                                              const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node), outputs, inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node), outputs, inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node), outputs, inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node), outputs, inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node), outputs, inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v3::Bucketize>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
