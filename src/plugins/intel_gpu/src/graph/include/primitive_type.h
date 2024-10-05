// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "openvino/core/type.hpp"

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
struct ImplementationManager;

struct primitive_type {
    virtual ~primitive_type() = default;

    virtual std::shared_ptr<program_node> create_node(program& program,
                                                      const std::shared_ptr<primitive> prim) const = 0;
    virtual std::shared_ptr<primitive_inst> create_instance(network& network,
                                                            const program_node& node) const = 0;

    virtual std::unique_ptr<primitive_impl> create_impl(const program_node& node) const = 0;
    virtual std::shared_ptr<ImplementationManager> choose_impl(const program_node& node, shape_types shape_type) const = 0;

    virtual std::set<impl_types> get_available_impl_types(const program_node& node) const = 0;
    virtual std::vector<std::shared_ptr<ImplementationManager>> get_supported_implementations(const program_node& node) const = 0;
    virtual const std::vector<std::shared_ptr<ImplementationManager>>& get_all_implementations() const = 0;
    virtual bool has_impl_for(const cldnn::program_node& node) const = 0;
    virtual bool has_impl_for(const cldnn::program_node& node, shape_types shape_type) const = 0;
    virtual bool has_impl_for(const cldnn::program_node& node, impl_types impl_type) const = 0;
    virtual bool has_impl_for(const cldnn::program_node& node, impl_types impl_type, shape_types shape_type) const = 0;
    virtual std::shared_ptr<ImplementationManager> get_best_impl(impl_types requested_impl_type, shape_types requested_shape_type) const = 0;
    virtual std::shared_ptr<ImplementationManager> get(const ov::DiscreteTypeInfo& type_info) const = 0;

    using in_out_fmts_t = std::pair<std::vector<format::type>, std::vector<format::type>>;
    virtual in_out_fmts_t query_preferred_formats(const cldnn::program_node& node, impl_types impl_type) const = 0;

    virtual layout calc_output_layout(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual std::vector<layout> calc_output_layouts(const program_node& node, const kernel_impl_params& impl_param) const = 0;
    virtual kernel_impl_params get_fake_aligned_params(kernel_impl_params const& orig_impl_param) const = 0;
    virtual std::string to_string(const program_node& node) const = 0;
};
}  // namespace cldnn
