// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/assign.hpp"
#include "primitive_inst.h"
#include "variable.hpp"

namespace cldnn {
template <>
struct typed_program_node<assign> : public typed_program_node_base<assign> {
private:
    using parent = typed_program_node_base<assign>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using assign_node = typed_program_node<assign>;

template<>
class typed_primitive_inst<assign> : public typed_primitive_inst_base<assign>, public memory_state::variable {
    using parent = typed_primitive_inst_base<assign>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(assign_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }

    static layout calc_output_layout(const assign_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const assign_node& node);

    typed_primitive_inst(network& network, const assign_node& desc);
    typed_primitive_inst(network& network) : parent(network), memory_state::variable("") {}

    void on_execute() override;
};

using assign_inst = typed_primitive_inst<assign>;

} // namespace cldnn
