// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/paged_attention.hpp"
#include "primitive_inst.h"

namespace cldnn {

using PagedAttentionInputIdx = cldnn::paged_attention::PagedAttentionInputIdx;

template <>
struct typed_program_node<paged_attention> : public typed_program_node_base<paged_attention> {
private:
    using parent = typed_program_node_base<paged_attention>;

public:
    using parent::parent;

    std::set<size_t> get_lockable_input_ids() const override {
        std::set<size_t> input_ports = { PagedAttentionInputIdx::PAST_LENS,
                                         PagedAttentionInputIdx::SUBSEQUENCE_BEGINS,
                                         PagedAttentionInputIdx::MAX_CONTEXT_LEN };

        if (typed_desc()->has_score_aggregation)
            input_ports.insert(PagedAttentionInputIdx::SCORE_AGGREGATION);

        return input_ports;
    }

    std::vector<size_t> get_shape_infer_dependencies() const override {
        const auto lockable_input_ids = get_lockable_input_ids();
        std::vector<size_t> input_ports(lockable_input_ids.begin(), lockable_input_ids.end());

        return input_ports;
    }
};

using paged_attention_node = typed_program_node<paged_attention>;

template<>
class typed_primitive_inst<paged_attention> : public typed_primitive_inst_base<paged_attention> {
    using parent = typed_primitive_inst_base<paged_attention>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(paged_attention_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const paged_attention_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const paged_attention_node& node);

    typed_primitive_inst(network& network, const paged_attention_node& desc);
    typed_primitive_inst(network& network) : parent(network) {}

    memory::ptr query_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::QUERY); }
    memory::ptr key_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::KEY); }
    memory::ptr value_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::VALUE); }
    memory::ptr key_cache_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::KEY_CACHE); }
    memory::ptr value_cache_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::VALUE_CACHE); }
    memory::ptr past_lens_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::PAST_LENS); }
    memory::ptr subsequence_begins_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::SUBSEQUENCE_BEGINS); }
    memory::ptr block_indices_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::BLOCK_INDICES); }
    memory::ptr block_indices_begins_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::BLOCK_INDICES_BEGINS); }
    memory::ptr alibi_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::ALIBI); }
    memory::ptr score_aggregation_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::SCORE_AGGREGATION); }
    memory::ptr rotated_block_indices_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::ROTATED_BLOCK_INDICES); }
    memory::ptr rotation_deltas_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::ROTATION_DELTAS); }
    memory::ptr rotation_trig_lut_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::ROTATION_TRIG_LUT); }
    memory::ptr xattention_threshold_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::XATTENTION_THRESHOLD); }
    memory::ptr xattention_block_size_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::XATTENTION_BLOCK_SIZE); }
    memory::ptr xattention_stride_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::XATTENTION_STRIDE); }
    memory::ptr sinks_memory_ptr() const { return input_memory_ptr(PagedAttentionInputIdx::SINKS); }
};

using paged_attention_inst = typed_primitive_inst<paged_attention>;

} // namespace cldnn
