// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gated_mlp_inst.h"
#include "impls/onednn/utils.hpp"
#include "intel_gpu/primitives/gated_mlp.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace cldnn {
namespace onednn {

#define LOG_AND_RETURN_FALSE_GATED(node) do {                                   \
    GPU_DEBUG_TRACE << (node).id() << " :  Do not select onednn" << std::endl; \
    return false;                                                                \
} while (0)

struct GatedMLPImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::gated_mlp")
    GatedMLPImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        if (!node.is_type<gated_mlp>())
            return false;

        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown || !config.get_use_onednn())
            LOG_AND_RETURN_FALSE_GATED(node);

        const auto& gm_node = node.as<gated_mlp>();
        const auto& gm_prim = gm_node.get_primitive();
        const auto& src_layout = gm_node.get_input_layout(0);
        const auto& out_layout = gm_node.get_output_layout(0);
        auto in0_dt = src_layout.data_type;
        auto out_dt = out_layout.data_type;

        // gate/up/down weight types (inputs 1, 2, 3)
        auto wei_gate_dt = gm_node.get_input_layout(1).data_type;
        auto wei_up_dt = gm_node.get_input_layout(2).data_type;
        auto wei_down_dt = gm_node.get_input_layout(3).data_type;

        if (one_of(data_types::i64, {in0_dt, wei_gate_dt, wei_up_dt, wei_down_dt}))
            LOG_AND_RETURN_FALSE_GATED(node);

        if (!everyone_is(format::bfyx, src_layout.format, out_layout.format) &&
            !everyone_is(format::bfzyx, src_layout.format, out_layout.format) &&
            !everyone_is(format::any, src_layout.format, out_layout.format))
            LOG_AND_RETURN_FALSE_GATED(node);

        if (!is_supported_pad(src_layout) || !is_supported_pad(out_layout))
            LOG_AND_RETURN_FALSE_GATED(node);

        // gated_mlp with compressed weights
        if (gm_prim->compressed_weights) {
            if (!one_of(in0_dt, {data_types::f16, data_types::f32, data_types::i8, data_types::u8}))
                LOG_AND_RETURN_FALSE_GATED(node);
            if (!one_of(wei_gate_dt, {data_types::u8, data_types::i8, data_types::u4, data_types::i4}))
                LOG_AND_RETURN_FALSE_GATED(node);
            if (!one_of(wei_up_dt, {data_types::u8, data_types::i8, data_types::u4, data_types::i4}))
                LOG_AND_RETURN_FALSE_GATED(node);
            if (!one_of(wei_down_dt, {data_types::u8, data_types::i8, data_types::u4, data_types::i4}))
                LOG_AND_RETURN_FALSE_GATED(node);
            if (!one_of(out_dt, {data_types::f16, data_types::f32}))
                LOG_AND_RETURN_FALSE_GATED(node);

            // Validate decompression zero point types
            if (gm_prim->decompression_zero_point_gate.is_valid()) {
                // zp_gate/up/down are at indices 7, 8, 9
                auto zp_gate_dt = gm_node.get_input_layout(7).data_type;
                if (!one_of(zp_gate_dt, {data_types::u8, data_types::i8, data_types::u4, data_types::i4}))
                    LOG_AND_RETURN_FALSE_GATED(node);
            }
        } else {
            // Non-compressed: f16 or f32 only
            if (!one_of(in0_dt, {data_types::f16, data_types::f32}))
                LOG_AND_RETURN_FALSE_GATED(node);
            if (!everyone_is(in0_dt, wei_gate_dt, wei_up_dt, wei_down_dt))
                LOG_AND_RETURN_FALSE_GATED(node);
            if (!one_of(out_dt, {data_types::f16, data_types::f32}))
                LOG_AND_RETURN_FALSE_GATED(node);
        }

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;
            in_fmts[idx] = format::get_default_format(out_rank);
        }
        out_fmts[0] = format::get_default_format(out_rank);

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
