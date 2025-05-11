// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/space_to_depth.hpp"
#include "primitive_inst.h"

#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<space_to_depth> : public typed_program_node_base<space_to_depth> {
    using parent = typed_program_node_base<space_to_depth>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using space_to_depth_node = typed_program_node<space_to_depth>;

template <>
class typed_primitive_inst<space_to_depth> : public typed_primitive_inst_base<space_to_depth> {
    using parent = typed_primitive_inst_base<space_to_depth>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(space_to_depth_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(space_to_depth_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(space_to_depth_node const& node);

public:
    typed_primitive_inst(network& network, space_to_depth_node const& desc);
};

using space_to_depth_inst = typed_primitive_inst<space_to_depth>;
}  // namespace cldnn
