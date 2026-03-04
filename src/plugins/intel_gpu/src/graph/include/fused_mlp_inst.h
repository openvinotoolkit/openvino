// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/primitives/fused_mlp.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<fused_mlp> : public typed_program_node_base<fused_mlp> {
private:
    using parent = typed_program_node_base<fused_mlp>;

public:
    using parent::parent;
    typed_program_node(std::shared_ptr<fused_mlp> prim, program& prog) : parent(prim, prog) {}

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts,
                                                               const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        return params;
    }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using fused_mlp_node = typed_program_node<fused_mlp>;

template <>
class typed_primitive_inst<fused_mlp> : public typed_primitive_inst_base<fused_mlp> {
    using parent = typed_primitive_inst_base<fused_mlp>;
    using parent::parent;
    using primitive_inst::update_output_memory;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const fused_mlp_node& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const fused_mlp_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const fused_mlp_node& node);
    typed_primitive_inst(network& network, const fused_mlp_node& node);
};

using fused_mlp_inst = typed_primitive_inst<fused_mlp>;

}  // namespace cldnn

