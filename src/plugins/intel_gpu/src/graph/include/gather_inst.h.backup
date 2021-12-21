// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/gather.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<gather> : public typed_program_node_base<gather> {
    using parent = typed_program_node_base<gather>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using gather_node = typed_program_node<gather>;

template <>
class typed_primitive_inst<gather> : public typed_primitive_inst_base<gather> {
    using parent = typed_primitive_inst_base<gather>;

public:
    static layout calc_output_layout(gather_node const& node);
    static std::string to_string(gather_node const& node);

public:
    typed_primitive_inst(network& network, gather_node const& desc);
};

using gather_inst = typed_primitive_inst<gather>;
}  // namespace cldnn
