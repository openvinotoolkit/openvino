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

/// @brief Validates partial MoE weight buffers after cached graph dependencies and outputs are restored.
void validate_moe_partial_buffers(const program& prog);

template <>
struct typed_program_node<moe_3gemm_fused_compressed> : public typed_program_node_base<moe_3gemm_fused_compressed> {
private:
    using parent = typed_program_node_base<moe_3gemm_fused_compressed>;

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<moe_3gemm_fused_compressed> prim, program& prog) : parent(prim, prog) {}

    /// @brief Checks that a partial buffer is a routed input compatible with this node's offload configuration.
    void validate_partial_buffer_input(const layout& buffer_layout, size_t resident_experts, size_t input_idx, const primitive_id& buffer_id) const;

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
    cldnn::moe_weights _weights;
    memory::ptr _expert_index_buffer;
};

using moe_inst = typed_primitive_inst<moe_3gemm_fused_compressed>;
}  // namespace cldnn
