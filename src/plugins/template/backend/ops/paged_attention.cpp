// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/paged_attention.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"

template <ov::element::Type_t ET>
bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    ov::reference::paged_attention<T>(outputs[0].data<T>(),
                                      outputs[1].data<T>(),
                                      outputs[2].data<T>(),
                                      outputs[3].data<T>(),
                                      inputs[0].data<T>(),         // q
                                      inputs[1].data<T>(),         // k
                                      inputs[2].data<T>(),         // v
                                      inputs[3].data<T>(),         // kc
                                      inputs[4].data<T>(),         // vc
                                      inputs[5].data<int32_t>(),   // pl
                                      inputs[6].data<int32_t>(),   // sb
                                      inputs[7].data<int32_t>(),   // bi
                                      inputs[8].data<int32_t>(),   // bib
                                      inputs[9].data<T>(),         // sc --
                                      inputs[10].data<int32_t>(),  // sw --
                                      inputs[11].data<T>(),        // as
                                      inputs[12].data<int32_t>(),  // mcl --
                                      inputs[13].get_shape().size() != 0 ? inputs[13].data<int32_t>() : nullptr,  // rbi
                                      inputs[14].get_shape().size() != 0 ? inputs[14].data<int32_t>() : nullptr,  // rd
                                      inputs[15].get_shape().size() != 0 ? inputs[15].data<T>() : nullptr,        // trl
                                      inputs[16].data<int32_t>(),                                                 // fbi
                                      inputs[17].data<int32_t>(),                                                 // mb
                                      inputs[0].get_shape(),                                                      // qs
                                      inputs[1].get_shape(),                                                      // ks
                                      inputs[2].get_shape(),                                                      // vs
                                      inputs[3].get_shape(),                                                      // kcs
                                      inputs[4].get_shape(),                                                      // vcs
                                      inputs[5].get_shape(),                                                      // pls
                                      inputs[13].get_shape(),  // rbis
                                      inputs[14].get_shape(),  // rds
                                      inputs[15].get_shape()   // rtls
    );
    return true;
}

template <>
bool evaluate_node<ov::op::PagedAttentionExtension>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type.get_type_name(), " in evaluate_node()");
    }
    return true;
}
