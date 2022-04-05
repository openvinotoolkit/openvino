// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gather_tree.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<gather_tree> : typed_program_node_base<gather_tree> {
private:
    using parent = typed_program_node_base<gather_tree>;
public:
    using parent::parent;
    typed_program_node(const std::shared_ptr<gather_tree> prim, program& prog) : parent(prim, prog) {
    }
    program_node& input() const { return get_dependency(0); }
};

using gather_tree_node = typed_program_node<gather_tree>;

template <>
class typed_primitive_inst<gather_tree> : public typed_primitive_inst_base<gather_tree> {
    using parent = typed_primitive_inst_base<gather_tree>;

public:
    static layout calc_output_layout(gather_tree_node const& node);
    static std::string to_string(gather_tree_node const& node);
    typed_primitive_inst(network& network, gather_tree_node const& node);
};

using gather_tree_inst = typed_primitive_inst<gather_tree>;

}  // namespace cldnn
