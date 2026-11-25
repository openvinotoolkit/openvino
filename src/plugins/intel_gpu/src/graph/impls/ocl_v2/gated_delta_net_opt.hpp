// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "gated_delta_net_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// GatedDeltaNet for the stateful (non-paged) op, recurrent state laid out as [B, H, K, V] (V innermost).
// Two stages chosen at runtime from the sequence length (see gated_delta_net_ref.cpp):
//  - seq_len <= max_decode_seq_len (1): gated_delta_net_opt.cl, one independent sub-group per (b, h, 16 V-columns),
//    state streamed with coalesced row accesses (no SLM / barriers / cross-lane reductions). The reference kernel
//    assigns 8 K-rows x 4 V-columns per lane, i.e. scalar loads strided by V plus 2 sub-group reductions per column.
//  - longer sequences: the reference kernel (state kept in registers across tokens).
struct GatedDeltaNetOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::gated_delta_net::opt")
    explicit GatedDeltaNetOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    static constexpr size_t subgroup_size = 16;
    // Measured on Arc 140T (48 layers, cold states): 1.4x faster than ref for 1 token, equal for 2, slower for 4.
    static constexpr int64_t max_decode_seq_len = 1;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<gated_delta_net>());
        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_types = {
            ov::element::f16,
            ov::element::f32,
        };

        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            if (!one_of(in_layout.format, supported_fmts) || !one_of(in_layout.data_type, supported_types)) {
                return false;
            }
        }
        for (size_t i = 0; i < node.get_outputs_count(); i++) {
            const auto& out_layout = node.get_output_layout(i);
            if (!one_of(out_layout.format, supported_fmts) || !one_of(out_layout.data_type, supported_types)) {
                return false;
            }
        }

        const auto& info = node.get_program().get_engine().get_device_info();
        if (!one_of(subgroup_size, info.supported_simd_sizes)) {
            return false;
        }

        // Head counts / head sizes must be static: they are baked into the kernel as JIT constants.
        const auto& q_shape = node.get_input_layout(0).get_partial_shape();
        const auto& v_shape = node.get_input_layout(2).get_partial_shape();
        if (q_shape.rank().is_dynamic() || v_shape.rank().is_dynamic() || q_shape.rank().get_length() != 4 || v_shape.rank().get_length() != 4 ||
            q_shape[2].is_dynamic() || q_shape[3].is_dynamic() || v_shape[2].is_dynamic() || v_shape[3].is_dynamic()) {
            return false;
        }
        const auto k_head_dims = static_cast<size_t>(q_shape[3].get_length());
        const auto v_head_dims = static_cast<size_t>(v_shape[3].get_length());
        const auto k_heads = static_cast<size_t>(q_shape[2].get_length());
        const auto v_heads = static_cast<size_t>(v_shape[2].get_length());
        // The opt kernel keeps K / 16 floats per lane for q, k and the partial sums: bound K to keep them in registers.
        return k_head_dims % subgroup_size == 0 && k_head_dims <= 256 && v_head_dims % subgroup_size == 0 && k_heads > 0 && v_heads % k_heads == 0;
    }
};

}  // namespace ov::intel_gpu::ocl
