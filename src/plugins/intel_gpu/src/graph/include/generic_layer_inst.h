// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "generic_layer.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<generic_layer> : public typed_program_node_base<generic_layer> {
    using parent = typed_program_node_base<generic_layer>;
    typed_program_node(const std::shared_ptr<generic_layer> prim, program& prog);

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
};

using generic_layer_node = typed_program_node<generic_layer>;

template <>
class typed_primitive_inst<generic_layer> : public typed_primitive_inst_base<generic_layer> {
    using parent = typed_primitive_inst_base<generic_layer>;
    using parent::parent;

public:
    static layout calc_output_layout(generic_layer_node const& node, kernel_impl_params const& impl_param) {
        return impl_param.typed_desc<generic_layer>()->output_layout;
    }

    static std::string to_string(generic_layer_node const& node);

    typed_primitive_inst(network& network, generic_layer_node const& node);
};

using generic_layer_inst = typed_primitive_inst<generic_layer>;

}  // namespace cldnn
