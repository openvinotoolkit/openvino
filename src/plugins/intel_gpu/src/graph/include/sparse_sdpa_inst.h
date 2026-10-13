// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_gpu/primitives/sparse_sdpa.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<sparse_sdpa> : public typed_program_node_base<sparse_sdpa> {
    using parent = typed_program_node_base<sparse_sdpa>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
};
using sparse_sdpa_node = typed_program_node<sparse_sdpa>;

template <>
class typed_primitive_inst<sparse_sdpa> : public typed_primitive_inst_base<sparse_sdpa> {
    using parent = typed_primitive_inst_base<sparse_sdpa>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const sparse_sdpa_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const sparse_sdpa_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const sparse_sdpa_node& node);
    typed_primitive_inst(network& network, const sparse_sdpa_node& node);
};

using sparse_sdpa_inst = typed_primitive_inst<sparse_sdpa>;

}  // namespace cldnn
