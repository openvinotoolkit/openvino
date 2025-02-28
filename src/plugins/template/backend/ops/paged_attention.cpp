// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/paged_attention.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v16::PagedAttention>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    ov::reference::paged_attention(outputs[0].data<ET>(),
                                   outputs[1].data<ET>(),
                                   inputs[0].data<ET>(),        // q
                                   inputs[1].data<ET>(),        // k
                                   inputs[2].data<ET>(),        // v
                                   inputs[3].data<ET>(),        // kc
                                   inputs[4].data<ET>(),        // vc
                                   inputs[0].get_shape(),       // qs
                                   inputs[1].get_shape(),       // kvs
                                   inputs[3].get_shape(),       // kvcs
                                   inputs[5].data<int32_t>(),   // pl
                                   inputs[6].data<int32_t>(),   // sb
                                   inputs[7].data<int32_t>(),   // bi
                                   inputs[8].data<int32_t>(),   // bib
                                   inputs[9].data<ET>(),        // sc --
                                   inputs[10].data<int32_t>(),  // sw --
                                   inputs[11].data<ET>(),       // as
                                   inputs[12].data<int32_t>(),  // mcl --
                                   inputs[13].data<int32_t>(),  // rbi
                                   inputs[14].data<int32_t>(),  // rd
                                   inputs[15].data<int32_t>(),  // trl
                                   inputs[13].get_shape(),      // kvcs
                                   inputs[14].get_shape(),      // kvcs
                                   inputs[15].get_shape());     // kvcs
    return true;
}

template <>
bool evaluate_node<ov::op::v16::PagedAttention>(std::shared_ptr<ov::Node> node,
                                                ov::TensorVector& outputs,
                                                const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v16::PagedAttention>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v16::PagedAttention>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v16::PagedAttention>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v16::PagedAttention>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
    return true;
}
