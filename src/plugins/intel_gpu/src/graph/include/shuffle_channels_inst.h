// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/shuffle_channels.hpp"
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
    typed_primitive_inst(network& network, shuffle_channels_node const& desc);
};

using shuffle_channels_inst = typed_primitive_inst<shuffle_channels>;
}  // namespace cldnn
