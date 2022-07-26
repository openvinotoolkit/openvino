// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "assign_inst.h"
#include "intel_gpu/primitives/read_value.hpp"
#include "primitive_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {

template<>
struct typed_program_node<read_value> : public typed_program_node_base<read_value> {
    using parent = typed_program_node_base<read_value>;
public:
    using parent::parent;

    const program_node& input(std::size_t index = 0) const { return get_dependency(index); }
};

using read_value_node = typed_program_node<read_value>;

template<>
class typed_primitive_inst<read_value> : public typed_primitive_inst_base<read_value>, public memory_state::variable {
    using parent = typed_primitive_inst_base<read_value>;

public:
    static layout calc_output_layout(const read_value_node& node);

    static std::string to_string(const read_value_node& node);

    typed_primitive_inst(network& network, const read_value_node& desc);
};

using read_value_inst = typed_primitive_inst<read_value>;

} // namespace cldnn
