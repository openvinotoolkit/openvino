// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/paged_attention.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/paged_cache_manager.hpp"
#include "openvino/core/type/element_iterator.hpp"

template <ov::element::Type_t ET>
bool evaluate(ov::TensorVector& outputs,
              const ov::TensorVector& inputs,
              std::shared_ptr<ov::Node> node,
              const std::shared_ptr<ov::internal::PagedCacheManager> cache_manager) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const bool has_rotation = inputs.size() == 16;
    const int rot = has_rotation ? 13 : -1;

    ov::reference::paged_attention<T>(node,
                                      cache_manager,
                                      outputs[0].data<T>(),
                                      outputs[1].data<T>(),
                                      inputs[0].data<T>(),                                       // q
                                      inputs[1].data<T>(),                                       // k
                                      inputs[2].data<T>(),                                       // v
                                      inputs[3].data<T>(),                                       // kc
                                      inputs[4].data<T>(),                                       // vc
                                      inputs[5].data<int32_t>(),                                 // pl
                                      inputs[6].data<int32_t>(),                                 // sb
                                      inputs[7].data<int32_t>(),                                 // bi
                                      inputs[8].data<int32_t>(),                                 // bib
                                      inputs[9].data<T>(),                                       // sc
                                      inputs[10].data<int32_t>(),                                // sw
                                      inputs[11].data<T>(),                                      // as
                                      inputs[12].data<int32_t>(),                                // mcl
                                      has_rotation ? inputs[rot + 0].data<int32_t>() : nullptr,  // rbi
                                      has_rotation ? inputs[rot + 1].data<int32_t>() : nullptr,  // rd
                                      has_rotation ? inputs[rot + 2].data<T>() : nullptr,        // trl
                                      inputs[0].get_shape(),                                     // qs
                                      inputs[1].get_shape(),                                     // ks
                                      inputs[2].get_shape(),                                     // vs
                                      inputs[3].get_shape(),                                     // kcs
                                      inputs[4].get_shape(),                                     // vcs
                                      inputs[5].get_shape(),                                     // pls
                                      has_rotation ? inputs[rot + 0].get_shape() : ov::Shape{},  // rbis
                                      has_rotation ? inputs[rot + 1].get_shape() : ov::Shape{},  // rds
                                      has_rotation ? inputs[rot + 2].get_shape() : ov::Shape{}   // rtls
    );
    return true;
}

template <>
bool evaluate_node<ov::op::PagedAttentionExtension>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    auto pa = std::static_pointer_cast<ov::op::PagedAttentionExtension>(node);
    auto cache_manager = pa->get_cache_manager();
    cache_manager->register_operator(pa);

    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(outputs, inputs, node, cache_manager);
    case ov::element::f16:
        return evaluate<ov::element::f16>(outputs, inputs, node, cache_manager);
    case ov::element::f64:
        return evaluate<ov::element::f64>(outputs, inputs, node, cache_manager);
    case ov::element::f32:
        return evaluate<ov::element::f32>(outputs, inputs, node, cache_manager);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
    return true;
}
