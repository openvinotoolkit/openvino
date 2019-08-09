/*
// Copyright (c) 2019 Intel Corporation
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
#include "api/CPP/shuffle_channels.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<shuffle_channels> : public typed_program_node_base<shuffle_channels> {
    using parent = typed_program_node_base<shuffle_channels>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using shuffle_channels_node = typed_program_node<shuffle_channels>;

template <>
class typed_primitive_inst<shuffle_channels> : public typed_primitive_inst_base<shuffle_channels> {
    using parent = typed_primitive_inst_base<shuffle_channels>;

public:
    static layout calc_output_layout(shuffle_channels_node const& node);
    static std::string to_string(shuffle_channels_node const& node);

public:
    typed_primitive_inst(network_impl& network, shuffle_channels_node const& desc);
};

using shuffle_channels_inst = typed_primitive_inst<shuffle_channels>;
}  // namespace cldnn
