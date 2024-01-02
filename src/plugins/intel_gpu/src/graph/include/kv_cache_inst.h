// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/kv_cache.hpp"
#include "primitive_inst.h"
#include "variable.hpp"

namespace cldnn {

template <>
struct typed_program_node<kv_cache> : public typed_program_node_base<kv_cache> {
private:
    using parent = typed_program_node_base<kv_cache>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using kv_cache_node = typed_program_node<kv_cache>;

template<>
class typed_primitive_inst<kv_cache> : public typed_primitive_inst_base<kv_cache>, public memory_state::variable {
    using parent = typed_primitive_inst_base<kv_cache>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(kv_cache_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const kv_cache_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const kv_cache_node& node);

    typed_primitive_inst(network& network, const kv_cache_node& desc);
    typed_primitive_inst(network& network) : parent(network), memory_state::variable("") {}
};

using kv_cache_inst = typed_primitive_inst<kv_cache>;

} // namespace cldnn
