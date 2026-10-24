// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

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
};
using qsa_indexer_node = typed_program_node<qsa_indexer>;

template <>
class typed_primitive_inst<qsa_indexer> : public typed_primitive_inst_base<qsa_indexer> {
    using parent = typed_primitive_inst_base<qsa_indexer>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const qsa_indexer_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const qsa_indexer_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const qsa_indexer_node& node);
    typed_primitive_inst(network& network, const qsa_indexer_node& node);
};

using qsa_indexer_inst = typed_primitive_inst<qsa_indexer>;

}  // namespace cldnn
