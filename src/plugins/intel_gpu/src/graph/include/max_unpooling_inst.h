// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/max_unpooling.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<max_unpooling> : public typed_program_node_base<max_unpooling> {
    using parent = typed_program_node_base<max_unpooling>;
    typed_program_node(const std::shared_ptr<max_unpooling> prim, program& prog);

public:
    using parent::parent;
    program_node& input() const { return get_dependency(0); }
    program_node& argmax() const { return get_dependency(1); }
};

using max_unpooling_node = typed_program_node<max_unpooling>;

template <>
class typed_primitive_inst<max_unpooling> : public typed_primitive_inst_base<max_unpooling> {
    using parent = typed_primitive_inst_base<max_unpooling>;

public:
    typed_primitive_inst(network& network, max_unpooling_node const& desc);
    static layout calc_output_layout(max_unpooling_node const& node);
    static std::string to_string(max_unpooling_node const& node);
};

using max_unpooling_inst = typed_primitive_inst<max_unpooling>;

}  // namespace cldnn
