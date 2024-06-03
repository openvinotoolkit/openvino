// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/rope.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<rope> : public typed_program_node_base<rope> {
    using parent = typed_program_node_base<rope>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using rope_node = typed_program_node<rope>;

template <>
class typed_primitive_inst<rope> : public typed_primitive_inst_base<rope> {
    using parent = typed_primitive_inst_base<rope>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const rope_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(rope_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(rope_node const& node);
};

using rope_inst = typed_primitive_inst<rope>;
}  // namespace cldnn
