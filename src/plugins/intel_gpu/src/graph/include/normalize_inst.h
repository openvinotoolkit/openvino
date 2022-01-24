// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/normalize.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<normalize> : public typed_program_node_base<normalize> {
    using parent = typed_program_node_base<normalize>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& scale() const { return get_dependency(1); }
};

using normalize_node = typed_program_node<normalize>;

template <>
class typed_primitive_inst<normalize> : public typed_primitive_inst_base<normalize> {
    using parent = typed_primitive_inst_base<normalize>;

public:
    static layout calc_output_layout(normalize_node const& node);
    static std::string to_string(normalize_node const& node);

public:
    typed_primitive_inst(network& network, normalize_node const& node);

    memory::ptr scale_memory() const { return dep_memory_ptr(1); }
};

using normalize_inst = typed_primitive_inst<normalize>;

}  // namespace cldnn
