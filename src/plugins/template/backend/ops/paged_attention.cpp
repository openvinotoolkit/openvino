// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/paged_attention.hpp"

#include <cstring>

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/reference/paged_attention.hpp"
#include "openvino/reference/utils/paged_cache_manager.hpp"

namespace {

template <ov::element::Type_t ET>
bool evaluate(ov::TensorVector& outputs,
              const ov::TensorVector& inputs,
              std::uintptr_t node_key,
              ov::reference::paged_attention_cache::PagedCacheManager* cache_manager) {
    using T = typename ov::element_type_traits<ET>::value_type;

    OPENVINO_ASSERT(inputs.size() == 25, "PagedAttentionExtension: expected 25 inputs");
    OPENVINO_ASSERT(outputs.size() == 3, "PagedAttentionExtension: expected 3 outputs");

    // Third output is currently shape-infer only
    // so I'm keeping it as 0 filled for determinism
    if (outputs[2].get_byte_size() > 0) {
        std::memset(outputs[2].data(), 0, outputs[2].get_byte_size());
    }

    ov::reference::paged_attention<T>(node_key,
                                      cache_manager,
                                      outputs[0].data<T>(),
                                      outputs[1].data<T>(),
                                      outputs[2].data<T>(),
                                      inputs[0].data<T>(),        // query
                                      inputs[1].data<T>(),        // key
                                      inputs[2].data<T>(),        // value
                                      inputs[3].data<T>(),        // key_cache init
                                      inputs[4].data<T>(),        // value_cache init
                                      inputs[5].data<int32_t>(),  // past_lens
                                      inputs[6].data<int32_t>(),  // subsequence_begins
                                      inputs[7].data<int32_t>(),  // block_indices init
                                      inputs[7].get_size(),
                                      inputs[8].data<int32_t>(),  // block_indices_begins init
                                      inputs[8].get_size(),
                                      inputs[9].data(),  // scale
                                      inputs[9].get_element_type(),
                                      inputs[10].data<int32_t>(),  // sliding_window
                                      inputs[11].data(),           // alibi_slopes
                                      inputs[11].get_element_type(),
                                      inputs[11].get_shape(),
                                      inputs[12].data<int32_t>(),  // max_context_len
                                      inputs[13].data<int32_t>(),  // score_aggregation_window
                                      inputs[14].data<int32_t>(),  // rotated_block_indices
                                      inputs[14].get_size(),
                                      inputs[15].data<int32_t>(),  // rotation_deltas
                                      inputs[15].get_shape(),
                                      inputs[16].data(),  // rotation_trig_lut
                                      inputs[16].get_element_type(),
                                      inputs[16].get_shape(),
                                      inputs[17].data(),  // xattention_threshold
                                      inputs[17].get_element_type(),
                                      inputs[18].data<int32_t>(),  // xattention_block_size
                                      inputs[19].data<int32_t>(),  // xattention_stride
                                      inputs[20].data(),           // sinks
                                      inputs[20].get_element_type(),
                                      inputs[21].data<int32_t>(),  // adaptive_rkv_start_size
                                      inputs[22].data<int32_t>(),  // adaptive_rkv_evictable_sizes
                                      inputs[23].data<int32_t>(),  // adaptive_rkv_diversity_block_set_indices
                                      inputs[24].data<int32_t>(),  // adaptive_rkv_diversity_block_set_indices_begins
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      inputs[2].get_shape(),
                                      inputs[3].get_shape(),
                                      inputs[4].get_shape(),
                                      inputs[5].get_shape(),
                                      inputs[6].get_shape());
    return true;
}

}  // namespace

template <>
bool evaluate_node<ov::op::PagedAttentionExtension>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs) {
    const auto& pa = std::static_pointer_cast<ov::op::PagedAttentionExtension>(node);
    const auto& handle = pa->get_cache_manager();
    auto* cache_manager = static_cast<ov::reference::paged_attention_cache::PagedCacheManager*>(handle.get());
    OPENVINO_ASSERT(cache_manager != nullptr, "PagedAttentionExtension: cache manager handle is null");

    const std::uintptr_t node_key = reinterpret_cast<std::uintptr_t>(node.get());
    const auto et = node->get_output_element_type(0);

    switch (et) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(outputs, inputs, node_key, cache_manager);
    case ov::element::f16:
        return evaluate<ov::element::f16>(outputs, inputs, node_key, cache_manager);
    case ov::element::f32:
        return evaluate<ov::element::f32>(outputs, inputs, node_key, cache_manager);
    case ov::element::f64:
        return evaluate<ov::element::f64>(outputs, inputs, node_key, cache_manager);
    default:
        OPENVINO_THROW("PagedAttentionExtension: unsupported element type: ", et);
    }
}
