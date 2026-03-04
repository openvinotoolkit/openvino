// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include <cstring>
#include <iostream>
#include <numeric>

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/reference/paged_attention.hpp"
#include "openvino/reference/utils/paged_cache_manager.hpp"
#include "paged_attention_shape_inference.hpp"
#include "tensor_data_accessor.hpp"

namespace {

// Resize output tensors calling shape_infer() with a runtime tensor accessor,
// exactly like other TEMPLATE ops (e.g. multinomial, SDPA, STFT).
// Unlike most ops, PA keeps some inputs dynamic even after
// validate_nodes_and_infer_types(), so we build input_shapes from the actual
// runtime tensors rather than from the node's (possibly dynamic) input shapes.
void resize_pa_outputs(const ov::op::PagedAttentionExtension* op,
                       ov::TensorVector& outputs,
                       const ov::TensorVector& inputs) {
    std::vector<ov::PartialShape> input_shapes;
    input_shapes.reserve(inputs.size());
    for (const auto& t : inputs) {
        input_shapes.emplace_back(t.get_shape());
    }
    const auto out_shapes = ov::op::shape_infer(op, input_shapes, ov::make_tensor_accessor(inputs));
    OPENVINO_ASSERT(out_shapes.size() == outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (out_shapes[i].is_static()) {
            outputs[i].set_shape(out_shapes[i].to_shape());
        } else {
            // Diversity output (output 2) may stay dynamic when evictable_sizes
            // is absent or empty — default to shape {0}.
            outputs[i].set_shape(ov::Shape{0});
        }
    }
}

template <ov::element::Type_t ET>
bool evaluate(const ov::op::PagedAttentionExtension* pa_op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs,
              std::uintptr_t node_key,
              ov::reference::paged_attention_cache::PagedCacheManager* cache_manager) {
    using T = typename ov::element_type_traits<ET>::value_type;

    OPENVINO_ASSERT(inputs.size() == 25, "PagedAttentionExtension: expected 25 inputs");
    OPENVINO_ASSERT(outputs.size() == 3, "PagedAttentionExtension: expected 3 outputs");

    std::cerr << "[PA_KERNEL_DBG] TEMPLATE reference kernel entered, q_shape=[" << inputs[0].get_shape()[0] << ","
              << inputs[0].get_shape()[1] << "], past_lens=[";
    for (size_t i = 0; i < inputs[5].get_shape()[0]; ++i)
        std::cerr << (i ? "," : "") << inputs[5].data<int32_t>()[i];
    std::cerr << "]" << std::endl;

    // Ensure output tensors are large enough for the current inputs.
    resize_pa_outputs(pa_op, outputs, inputs);

    // Third output is currently shape-infer only
    // so I'm keeping it as 0 filled for determinism
    if (outputs[2].get_byte_size() > 0) {
        std::memset(outputs[2].data(), 0, outputs[2].get_byte_size());
    }

    // For optional inputs with Shape{0} (disabled), pass nullptr so the reference
    // can distinguish "absent" from "present but empty".
    const void* xattn_thresh_ptr =
        inputs[17].get_shape().empty() || inputs[17].get_size() == 0 ? nullptr : inputs[17].data();
    const auto xattn_thresh_et = (xattn_thresh_ptr != nullptr) ? inputs[17].get_element_type() : ov::element::dynamic;
    const void* sinks_ptr = inputs[20].get_shape().empty() || inputs[20].get_size() == 0 ? nullptr : inputs[20].data();
    const auto sinks_et = (sinks_ptr != nullptr) ? inputs[20].get_element_type() : ov::element::dynamic;

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
                                      xattn_thresh_ptr,  // xattention_threshold
                                      xattn_thresh_et,
                                      inputs[18].data<int32_t>(),  // xattention_block_size
                                      inputs[19].data<int32_t>(),  // xattention_stride
                                      sinks_ptr,                   // sinks
                                      sinks_et,
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
        return evaluate<ov::element::bf16>(pa.get(), outputs, inputs, node_key, cache_manager);
    case ov::element::f16:
        return evaluate<ov::element::f16>(pa.get(), outputs, inputs, node_key, cache_manager);
    case ov::element::f32:
        return evaluate<ov::element::f32>(pa.get(), outputs, inputs, node_key, cache_manager);
    case ov::element::f64:
        return evaluate<ov::element::f64>(pa.get(), outputs, inputs, node_key, cache_manager);
    default:
        OPENVINO_THROW("PagedAttentionExtension: unsupported element type: ", et);
    }
}
