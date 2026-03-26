// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gated_mlp_inst.h"
#include "impls/onednn/utils.hpp"
#include "intel_gpu/primitives/gated_mlp.hpp"
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
        if (!config.get_use_onednn() || !info.supports_immad)
            LOG_AND_RETURN_FALSE_GATED(node);

        const auto& gm_node = node.as<gated_mlp>();
        const auto& src_layout = gm_node.get_input_layout(0);
        const auto& out_layout = gm_node.get_output_layout(0);

        const auto src_rank = src_layout.get_partial_shape().rank();
        const auto out_rank = out_layout.get_partial_shape().rank();
        if ((src_rank.is_static() && src_rank.get_length() < 2) ||
            (out_rank.is_static() && out_rank.get_length() < 2))
            LOG_AND_RETURN_FALSE_GATED(node);

        if (!is_supported_pad(src_layout) || !is_supported_pad(out_layout))
            LOG_AND_RETURN_FALSE_GATED(node);

        if (src_layout.data_type != data_types::f16 && src_layout.data_type != data_types::f32)
            LOG_AND_RETURN_FALSE_GATED(node);

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);
        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
