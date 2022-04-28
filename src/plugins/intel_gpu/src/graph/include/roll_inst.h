// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/roll.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<roll> : public typed_program_node_base<roll> {
    using parent = typed_program_node_base<roll>;

public:
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }
};

using roll_node = typed_program_node<roll>;

template <>
class typed_primitive_inst<roll> : public typed_primitive_inst_base<roll> {
    using parent = typed_primitive_inst_base<roll>;

public:
    static layout calc_output_layout(const roll_node& node);
    static std::string to_string(const roll_node& node);

public:
    typed_primitive_inst(network& network, const roll_node& desc);
};

using roll_inst = typed_primitive_inst<roll>;

}  // namespace cldnn
