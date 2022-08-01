// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/assign.hpp"
#include "primitive_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace memory_state {

class variable {
public:
    explicit variable(const std::string& variable_id) : variable_id_ {variable_id} {}

    const std::string& variable_id() const { return variable_id_; }

private:
    std::string variable_id_;
};

} // namespace memory_state

template<>
struct typed_program_node<assign> : public typed_program_node_base<assign> {
    using parent = typed_program_node_base<assign>;
public:
    using parent::parent;

    const program_node& input(std::size_t index = 0) const { return get_dependency(index); }
};

using assign_node = typed_program_node<assign>;

template<>
class typed_primitive_inst<assign> : public typed_primitive_inst_base<assign>, public memory_state::variable {
    using parent = typed_primitive_inst_base<assign>;

public:
    static layout calc_output_layout(const assign_node& node);

    static std::string to_string(const assign_node& node);

public:
    typed_primitive_inst(network& network, const assign_node& desc);
};

using assign_inst = typed_primitive_inst<assign>;

} // namespace cldnn
