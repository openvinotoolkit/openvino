// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/activation.hpp"
#include "primitive_inst.h"
#include "kernel_selector/core/actual_kernels/activation/activation_kernel_base.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<activation> : public typed_program_node_base<activation> {
    using parent = typed_program_node_base<activation>;
    typed_program_node(const std::shared_ptr<activation> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& slope_input() const { return get_dependency(1); }

    bool is_parameterized() const { return !typed_desc()->additional_params_input.empty(); }

    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        kernel_selector::base_activation_params p;
        p.function = get_kernel_selector_activation_param(typed_desc()->activation_function);
        p.m = typed_desc()->additional_params.a;
        p.n = typed_desc()->additional_params.b;
        return std::make_shared<kernel_selector::activation_fuse_params>(p);
    }
};

using activation_node = typed_program_node<activation>;

template <>
class typed_primitive_inst<activation> : public typed_primitive_inst_base<activation> {
    using parent = typed_primitive_inst_base<activation>;

public:
    static layout calc_output_layout(activation_node const& node);
    static std::string to_string(activation_node const& node);

public:
    typed_primitive_inst(network& network, activation_node const& node);

    memory::ptr slope_memory() const { return dep_memory_ptr(1); }

    bool is_parameterized() const { return !argument.additional_params_input.empty(); }
};

using activation_inst = typed_primitive_inst<activation>;
}  // namespace cldnn
