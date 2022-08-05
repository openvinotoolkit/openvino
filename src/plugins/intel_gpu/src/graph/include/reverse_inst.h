// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>

#include "intel_gpu/primitives/reverse.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<reverse> : public typed_program_node_base<reverse> {
    using parent = typed_program_node_base<reverse>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
};

using reverse_node = typed_program_node<reverse>;

template <>
class typed_primitive_inst<reverse> : public typed_primitive_inst_base<reverse> {
    using parent = typed_primitive_inst_base<reverse>;

public:
    static layout calc_output_layout(reverse_node const& node);
    static std::string to_string(reverse_node const& node);

public:
    typed_primitive_inst(network& network, reverse_node const& desc);
};

using reverse_inst = typed_primitive_inst<reverse>;
}  // namespace cldnn
