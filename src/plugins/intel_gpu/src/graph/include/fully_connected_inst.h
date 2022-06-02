// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/fully_connected.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<fully_connected> : public typed_program_node_base<fully_connected> {
    using parent = typed_program_node_base<fully_connected>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog) {}

    program_node& input() const { return get_dependency(0); }
    program_node& weights() const { return get_dependency(1); }
    program_node& bias() const { return get_dependency(2); }
    bool bias_term() const { return !get_primitive()->bias.empty(); }
    std::shared_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts,
                                              const layout& out_layout) const override {
        const auto& bias_layout = bias_term() ?  bias().get_output_layout() : layout(data_types::f32, format::any, tensor());
        return std::make_shared<kernel_impl_params>(get_program(), get_primitive(), get_unique_id(),
                                  in_layouts, out_layout,
                                  get_fused_primitives(), get_fused_activations_funcs(), get_fused_activations_params(),
                                  weights().get_output_layout(), bias_term(), bias_layout);
    }
};

using fully_connected_node = typed_program_node<fully_connected>;

template <>
class typed_primitive_inst<fully_connected> : public typed_primitive_inst_base<fully_connected> {
    using parent = typed_primitive_inst_base<fully_connected>;

public:
    static layout calc_output_layout(fully_connected_node const& node);
    static std::string to_string(fully_connected_node const& node);

public:
    typed_primitive_inst(network& network, fully_connected_node const& node);

    memory::ptr weights_memory() const { return dep_memory_ptr(1); }
    memory::ptr bias_memory() const { return dep_memory_ptr(2); }

    bool bias_term() const { return !argument.bias.empty(); }
};

using fully_connected_inst = typed_primitive_inst<fully_connected>;

}  // namespace cldnn
