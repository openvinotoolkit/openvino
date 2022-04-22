// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/select.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<select> : public typed_program_node_base<select> {
    using parent = typed_program_node_base<select>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return *get_dependency(idx).first; }
    size_t inputs_count() const { return get_dependencies().size(); }
};

using select_node = typed_program_node<select>;

template <>
class typed_primitive_inst<select> : public typed_primitive_inst_base<select> {
    using parent = typed_primitive_inst_base<select>;

public:
    static layout calc_output_layout(select_node const& node);
    static std::string to_string(select_node const& node);
    typed_primitive_inst(network& network, select_node const& node);
};

using select_inst = typed_primitive_inst<select>;
}  // namespace cldnn
