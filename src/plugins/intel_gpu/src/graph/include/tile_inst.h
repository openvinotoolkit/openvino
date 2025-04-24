// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/tile.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<tile> : public typed_program_node_base<tile> {
    using parent = typed_program_node_base<tile>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1}; }
};

using tile_node = typed_program_node<tile>;

template <>
class typed_primitive_inst<tile> : public typed_primitive_inst_base<tile> {
    using parent = typed_primitive_inst_base<tile>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(tile_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(tile_node const& node, kernel_impl_params const& impl_param);

    static std::string to_string(tile_node const& node);

public:
    typed_primitive_inst(network& network, tile_node const& desc);
};

using tile_inst = typed_primitive_inst<tile>;

}  // namespace cldnn
