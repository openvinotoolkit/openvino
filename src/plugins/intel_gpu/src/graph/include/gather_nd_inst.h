// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gather_nd.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<gather_nd> : public typed_program_node_base<gather_nd> {
    using parent = typed_program_node_base<gather_nd>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using gather_nd_node = typed_program_node<gather_nd>;

template <>
class typed_primitive_inst<gather_nd> : public typed_primitive_inst_base<gather_nd> {
    using parent = typed_primitive_inst_base<gather_nd>;

public:
    static layout calc_output_layout(gather_nd_node const& node);
    static std::string to_string(gather_nd_node const& node);

public:
    typed_primitive_inst(network& network, gather_nd_node const& desc);
};

using gather_nd_inst = typed_primitive_inst<gather_nd>;
}  // namespace cldnn
