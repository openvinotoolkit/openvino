// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/search_sorted.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v15::SearchSorted>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::reference::search_sorted<T>(inputs[0].data<const T>(),
                                    inputs[1].data<const T>(),
                                    outputs[0].data<int64_t>(),
                                    op->get_input_shape(0),
                                    op->get_input_shape(1),
                                    op->get_right_mode());
    return true;
}

template <>
bool evaluate_node<ov::op::v15::SearchSorted>(std::shared_ptr<ov::Node> node,
                                              ov::TensorVector& outputs,
                                              const ov::TensorVector& inputs) {
    const auto& element_type = node->get_input_element_type(0);

#define CASE(type)          \
    case ov::element::type: \
        return evaluate<ov::element::type>(ov::as_type_ptr<ov::op::v15::SearchSorted>(node), outputs, inputs);

    switch (element_type) {
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(f64);
        CASE(i8);
        CASE(i16);
        CASE(i32);
        CASE(i64);
        CASE(u8);
        CASE(u16);
        CASE(u32);
        CASE(u64);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
#undef CASE
}
