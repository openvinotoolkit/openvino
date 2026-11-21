// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "experimental/primitives/draft_block_mask.hpp"
#include "experimental/registry/registry.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<draft_block_mask> : public typed_program_node_base<draft_block_mask> {
    using parent = typed_program_node_base<draft_block_mask>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        // topk (input 4, 5-input runtime-seqlen/topk variant only) is read by
        // DraftBlockMaskTopKGenerator's dispatch_data_func via mem_lock (read_topk_value()) --
        // registering it here is what populates kernel_impl_params::memory_deps for that read (see
        // vl_sdpa_inst.h's identical precedent for its cu_seq_lens input). seqlen (input 3) is NOT
        // needed here: the CM impl derives it from local_mask's own shape instead (see
        // resolve_geometry() in graph/impls/cm/draft_block_mask/draft_block_mask.cpp).
        if (get_dependencies().size() == 5) {
            return {4};
        }
        return {};
    }
};
using draft_block_mask_node = typed_program_node<draft_block_mask>;

template <>
class typed_primitive_inst<draft_block_mask> : public typed_primitive_inst_base<draft_block_mask> {
    using parent = typed_primitive_inst_base<draft_block_mask>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const draft_block_mask_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const draft_block_mask_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const draft_block_mask_node& node);
    typed_primitive_inst(network& network, const draft_block_mask_node& node);
};

using draft_block_mask_inst = typed_primitive_inst<draft_block_mask>;
}  // namespace cldnn
