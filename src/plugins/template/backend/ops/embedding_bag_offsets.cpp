// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/embedding_bag_offsets.hpp"

#include "evaluate_node.hpp"

namespace embedding_bag_offsets_v15 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v15::EmbeddingBagOffsets>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ov::reference::embeddingBagOffsets<T1, T2>(inputs[0].data<T1>(),
                                               inputs[1].data<T2>(),
                                               inputs[2].data<T2>(),
                                               inputs.size() > 3 ? inputs[3].data<T2>() : nullptr,
                                               inputs.size() > 4 ? inputs[4].data<T1>() : nullptr,
                                               outputs[0].data<T1>(),
                                               ov::shape_size(inputs[1].get_shape()),
                                               outputs[0].get_shape(),
                                               op->get_reduction());
}
}  // namespace embedding_bag_offsets_v15

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v15::EmbeddingBagOffsets>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (inputs[1].get_element_type()) {
    case ov::element::i32:
        embedding_bag_offsets_v15::evaluate<ET, ov::element::i32>(op, outputs, inputs);
        break;
    case ov::element::i64:
        embedding_bag_offsets_v15::evaluate<ET, ov::element::i64>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v15::EmbeddingBagOffsets>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
