// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

#include <memory>
#include <string>

namespace cldnn {
struct network;
class engine;
struct program_node;
struct primitive_impl;
class primitive_inst;
struct program;
struct primitive;

struct primitive_type {
    virtual ~primitive_type() = default;

    virtual std::shared_ptr<program_node> create_node(program& program,
                                                      const std::shared_ptr<primitive> prim) const = 0;
    virtual std::shared_ptr<primitive_inst> create_instance(network& network,
                                                            const program_node& node) const = 0;
    virtual std::shared_ptr<primitive_inst> create_instance(network& network) const = 0;

    virtual std::unique_ptr<primitive_impl> choose_impl(const program_node& node) const = 0;
    virtual std::unique_ptr<primitive_impl> choose_impl(const program_node& node, const kernel_impl_params& params) const = 0;

    virtual bool does_an_implementation_exist(const program_node& node) const = 0;
    virtual bool does_an_implementation_exist(const program_node& node, const kernel_impl_params& params) const = 0;

    virtual bool does_possible_implementation_exist(const program_node& node) const = 0;
    virtual bool does_possible_implementation_exist(const program_node& node, const kernel_impl_params& params) const = 0;

    virtual bool does_dynamic_implementation_exist(const program_node& node) const = 0;
    virtual bool does_dynamic_implementation_exist(const program_node& node, const kernel_impl_params& params) const = 0;

    virtual layout calc_output_layout(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual std::vector<layout> calc_output_layouts(const program_node& node, const kernel_impl_params& impl_param) const = 0;
    virtual kernel_impl_params get_fake_aligned_params(kernel_impl_params const& orig_impl_param) const = 0;
    virtual std::vector<size_t> extend_input_shape_to_6d(kernel_impl_params const& orig_impl_param, int32_t input_idx) const = 0;
    virtual std::vector<size_t> extend_output_shape_to_6d(kernel_impl_params const& orig_impl_param, int32_t output_idx) const = 0;
    virtual std::string to_string(const program_node& node) const = 0;
};
}  // namespace cldnn
