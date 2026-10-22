// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>

#include "intel_gpu/primitives/recurrent_cache_update.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<recurrent_cache_update> : public typed_program_node_base<recurrent_cache_update> {
    using parent = typed_program_node_base<recurrent_cache_update>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};
using recurrent_cache_update_node = typed_program_node<recurrent_cache_update>;

template <>
class typed_primitive_inst<recurrent_cache_update> : public typed_primitive_inst_base<recurrent_cache_update> {
    using parent = typed_primitive_inst_base<recurrent_cache_update>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const recurrent_cache_update_node& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const recurrent_cache_update_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const recurrent_cache_update_node& node);
    typed_primitive_inst(network& network, const recurrent_cache_update_node& node);
};

using recurrent_cache_update_inst = typed_primitive_inst<recurrent_cache_update>;
}  // namespace cldnn
