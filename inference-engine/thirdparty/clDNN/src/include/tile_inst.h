/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/CPP/tile.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

template <>
struct typed_program_node<tile> : public typed_program_node_base<tile> {
    using parent = typed_program_node_base<tile>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
};

using tile_node = typed_program_node<tile>;

template <>
class typed_primitive_inst<tile> : public typed_primitive_inst_base<tile> {
    using parent = typed_primitive_inst_base<tile>;

public:
    static layout calc_output_layout(tile_node const& node);
    static std::string to_string(tile_node const& node);

public:
    typed_primitive_inst(network_impl& network, tile_node const& desc);
};

using tile_inst = typed_primitive_inst<tile>;

}  // namespace cldnn
