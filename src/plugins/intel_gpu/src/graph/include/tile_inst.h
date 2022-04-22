// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
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

    program_node& input() const { return *get_dependency(0).first; }
};

using tile_node = typed_program_node<tile>;

template <>
class typed_primitive_inst<tile> : public typed_primitive_inst_base<tile> {
    using parent = typed_primitive_inst_base<tile>;

public:
    static layout calc_output_layout(tile_node const& node);
    static std::string to_string(tile_node const& node);

public:
    typed_primitive_inst(network& network, tile_node const& desc);
};

using tile_inst = typed_primitive_inst<tile>;

}  // namespace cldnn
