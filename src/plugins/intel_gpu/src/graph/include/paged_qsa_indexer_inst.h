// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "intel_gpu/primitives/paged_qsa_indexer.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<paged_qsa_indexer> : public typed_program_node_base<paged_qsa_indexer> {
    using parent = typed_program_node_base<paged_qsa_indexer>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }

    // The output shapes follow the index_query shape only, and the impl reads no input on the host (past_lens,
    // subsequence_begins and the page table are resolved by the kernels). The default would make every input a shape
    // infer dependency, so each execution would wait on the host for its producers, a per-layer synchronization.
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};
using paged_qsa_indexer_node = typed_program_node<paged_qsa_indexer>;

template <>
class typed_primitive_inst<paged_qsa_indexer> : public typed_primitive_inst_base<paged_qsa_indexer> {
    using parent = typed_primitive_inst_base<paged_qsa_indexer>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const paged_qsa_indexer_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const paged_qsa_indexer_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const paged_qsa_indexer_node& node);
    typed_primitive_inst(network& network, const paged_qsa_indexer_node& node);
};

using paged_qsa_indexer_inst = typed_primitive_inst<paged_qsa_indexer>;

}  // namespace cldnn
