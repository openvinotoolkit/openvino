// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gather_tree.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

using gather_tree_node = typed_program_node<gather_tree>;

template <>
class typed_primitive_inst<gather_tree> : public typed_primitive_inst_base<gather_tree> {
    using parent = typed_primitive_inst_base<gather_tree>;
    using parent::parent;

public:
    static layout calc_output_layout(gather_tree_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gather_tree_node const& node);
    typed_primitive_inst(network& network, gather_tree_node const& node);
};

using gather_tree_inst = typed_primitive_inst<gather_tree>;

}  // namespace cldnn
