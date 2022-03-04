// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/average_unpooling.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<average_unpooling> : public typed_program_node_base<average_unpooling> {
    using parent = typed_program_node_base<average_unpooling>;

public:
    using parent::parent;
    program_node& input() const { return get_dependency(0); }
};

using average_unpooling_node = typed_program_node<average_unpooling>;

template <>
class typed_primitive_inst<average_unpooling> : public typed_primitive_inst_base<average_unpooling> {
    using parent = typed_primitive_inst_base<average_unpooling>;

public:
    typed_primitive_inst(network& network, average_unpooling_node const& desc);
    static layout calc_output_layout(average_unpooling_node const& node);
    static std::string to_string(average_unpooling_node const& node);
};

using average_unpooling_inst = typed_primitive_inst<average_unpooling>;

}  // namespace cldnn
