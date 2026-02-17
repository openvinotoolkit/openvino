// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "primitive_inst.h"

namespace cldnn {
namespace details {}

template <>
struct typed_program_node<moe_3gemm_fused_compressed> : public typed_program_node_base<moe_3gemm_fused_compressed> {
private:
    using parent = typed_program_node_base<moe_3gemm_fused_compressed>;

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<moe_3gemm_fused_compressed> prim, program& prog) : parent(prim, prog) {}

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);

        return params;
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using moe_node = typed_program_node<moe_3gemm_fused_compressed>;

template <>
class typed_primitive_inst<moe_3gemm_fused_compressed> : public typed_primitive_inst_base<moe_3gemm_fused_compressed> {
    using parent = typed_primitive_inst_base<moe_3gemm_fused_compressed>;
    using parent::parent;
    using primitive_inst::update_output_memory;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const moe_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const moe_node& /* node */, const kernel_impl_params& impl_param);
    static std::string to_string(const moe_node& node);
    typed_primitive_inst(network& network, const moe_node& node);
};

using moe_inst = typed_primitive_inst<moe_3gemm_fused_compressed>;
}  // namespace cldnn
