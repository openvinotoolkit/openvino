// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/ctc_loss.hpp"

#include "evaluate_node.hpp"

namespace ctc_loss_v4 {
template <
    ov::element::Type_t ET1,
    ov::element::Type_t ET2,
    typename std::enable_if<!std::is_floating_point<typename ov::element_type_traits<ET1>::value_type>::value &&
                                !std::is_same<typename ov::element_type_traits<ET1>::value_type, ov::bfloat16>::value &&
                                !std::is_same<typename ov::element_type_traits<ET1>::value_type, ov::float16>::value,
                            bool>::type = true>
inline void evaluate(const std::shared_ptr<ov::op::v4::CTCLoss>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    OPENVINO_THROW("The data type for logits is expected to be a floating point type. Got:", ov::element::Type(ET1));
}

template <
    ov::element::Type_t ET1,
    ov::element::Type_t ET2,
    typename std::enable_if<std::is_floating_point<typename ov::element_type_traits<ET1>::value_type>::value ||
                                std::is_same<typename ov::element_type_traits<ET1>::value_type, ov::bfloat16>::value ||
                                std::is_same<typename ov::element_type_traits<ET1>::value_type, ov::float16>::value,
                            bool>::type = true>
inline void evaluate(const std::shared_ptr<ov::op::v4::CTCLoss>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<ET1>::value_type;
    using T2 = typename ov::element_type_traits<ET2>::value_type;
    ov::reference::CTCLoss<T1, T2>(static_cast<T1*>(inputs[0].data()),
                                   inputs[0].get_shape(),
                                   static_cast<T2*>(inputs[1].data()),
                                   static_cast<T2*>(inputs[2].data()),
                                   static_cast<T2*>(inputs[3].data()),
                                   static_cast<T2*>(inputs[4].data()),
                                   op->get_preprocess_collapse_repeated(),
                                   op->get_ctc_merge_repeated(),
                                   op->get_unique(),
                                   static_cast<T1*>(outputs[0].data()));
}
}  // namespace ctc_loss_v4

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v4::CTCLoss>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (inputs[1].get_element_type()) {
    case ov::element::i32:
        ctc_loss_v4::evaluate<ET, ov::element::i32>(op, outputs, inputs);
        break;
    case ov::element::i64:
        ctc_loss_v4::evaluate<ET, ov::element::i64>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v4::CTCLoss>(std::shared_ptr<ov::Node> node,
                                        ov::TensorVector& outputs,
                                        const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v4::CTCLoss>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
