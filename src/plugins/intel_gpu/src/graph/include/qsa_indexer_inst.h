// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <string>
#include <vector>

#include "intel_gpu/primitives/qsa_indexer.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<qsa_indexer> : public typed_program_node_base<qsa_indexer> {
    using parent = typed_program_node_base<qsa_indexer>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }

    // The block_index_keys_cache output has max(key_length) / r rows, so the output shape depends on the key_length
    // values and shape inference reads them on the host. As paged_attention does for past_lens, a key_length that is
    // a network input is uploaded blocking into lockable (host-visible USM) memory, so that read does not synchronize
    // the queue. This holds only when key_length is directly a network input: if another GPU primitive produces it,
    // the host read waits for that primitive to finish.
    std::set<size_t> get_lockable_input_ids() const override {
        return {qsa_indexer::InputIdx::KEY_LENGTH};
    }

    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {qsa_indexer::InputIdx::KEY_LENGTH};
    }
};
using qsa_indexer_node = typed_program_node<qsa_indexer>;

template <>
class typed_primitive_inst<qsa_indexer> : public typed_primitive_inst_base<qsa_indexer> {
    using parent = typed_primitive_inst_base<qsa_indexer>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const qsa_indexer_node& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const qsa_indexer_node& node, const kernel_impl_params& impl_param);

    static std::string to_string(const qsa_indexer_node& node);
    typed_primitive_inst(network& network, const qsa_indexer_node& node);
};

using qsa_indexer_inst = typed_primitive_inst<qsa_indexer>;

}  // namespace cldnn
