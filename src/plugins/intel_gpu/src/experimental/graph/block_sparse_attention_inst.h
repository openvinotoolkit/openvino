// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "experimental/primitives/block_sparse_attention.hpp"
#include "experimental/registry/registry.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<block_sparse_attention> : public typed_program_node_base<block_sparse_attention> {
    using parent = typed_program_node_base<block_sparse_attention>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        // scale (input 4, 5-input runtime-scale variant only) is read by
        // BlockSparseAttentionPrefillGenerator/DecodeGenerator's dispatch_data_func via mem_lock
        // (read_scale_value()) -- registering it here is what populates
        // kernel_impl_params::memory_deps for that read (see draft_block_mask_inst.h's identical
        // precedent for its topk input, itself citing vl_sdpa_inst.h's cu_seq_lens).
        if (get_dependencies().size() == 5) {
            return {4};
        }
        return {};
    }
};
using block_sparse_attention_node = typed_program_node<block_sparse_attention>;

template <>
class typed_primitive_inst<block_sparse_attention> : public typed_primitive_inst_base<block_sparse_attention> {
    using parent = typed_primitive_inst_base<block_sparse_attention>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const block_sparse_attention_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const block_sparse_attention_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const block_sparse_attention_node& node);
    typed_primitive_inst(network& network, const block_sparse_attention_node& node);
};

using block_sparse_attention_inst = typed_primitive_inst<block_sparse_attention>;
}  // namespace cldnn
