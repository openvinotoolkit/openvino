// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/arg_max_min.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<arg_max_min> : public typed_program_node_base<arg_max_min> {
    using parent = typed_program_node_base<arg_max_min>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog) : parent(prim, prog) {}
    program_node& input() const { return get_dependency(0); }
};

using arg_max_min_node = typed_program_node<arg_max_min>;

template <>
class typed_primitive_inst<arg_max_min> : public typed_primitive_inst_base<arg_max_min> {
    using parent = typed_primitive_inst_base<arg_max_min>;

public:
    static layout calc_output_layout(arg_max_min_node const& node);
    static std::string to_string(arg_max_min_node const& node);

public:
    typed_primitive_inst(network_impl& network, arg_max_min_node const& node);
};

using arg_max_min_inst = typed_primitive_inst<arg_max_min>;

}  // namespace cldnn
