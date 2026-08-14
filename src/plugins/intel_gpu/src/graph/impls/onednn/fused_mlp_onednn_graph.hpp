// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fused_mlp_inst.h"
#include "impls/onednn/utils.hpp"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <memory>

#define LOG_AND_RETURN_FALSE(node) do {                                         \
    GPU_DEBUG_TRACE << (node).id() << " :  Do not select onednn" << std::endl;  \
    return false;                                                               \
} while (0)

namespace cldnn {
namespace onednn {

struct FusedMLPImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn_graph::fused_mlp")
    FusedMLPImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    static bool try_extract_dims(const fused_mlp_node& node, int64_t& mb, int64_t& ic, int64_t& oc) {
        const auto& x_layout = node.get_input_layout(0);
        const auto& w_gate_layout = node.get_input_layout(1);
        const auto& w_up_layout = node.get_input_layout(2);
        const auto& w_down_layout = node.get_input_layout(3);

        if (w_gate_layout.is_dynamic() || w_up_layout.is_dynamic() || w_down_layout.is_dynamic())
            return false;

        auto w_gate = w_gate_layout.get_tensor().sizes(format::bfyx);
        auto w_up = w_up_layout.get_tensor().sizes(format::bfyx);
        auto w_down = w_down_layout.get_tensor().sizes(format::bfyx);

        const auto& x_pshape = x_layout.get_partial_shape();
        if (x_pshape.is_dynamic()) {
            if (!x_pshape.rank().is_static())
                return false;

            auto r = x_pshape.rank().get_length();
            if (r != 2 && r != 4)
                return false;
            
            // get IC：rank2 -> [1], rank4 -> [2]
            const auto& ic_dim = (r == 2) ? x_pshape[1] : x_pshape[2];
            if (!ic_dim.is_static())
                return false;
            ic = ic_dim.get_length();

            if (r == 4 && !x_pshape[3].compatible(1))
                return false;
        } else {
            auto x = x_layout.get_tensor().sizes(format::bfyx);          // [b, f, y, x] in bfyx order// X supported as:
            
            //   rank2  -> [MB, IC] encoded as [b=MB, f=IC, y=1, x=1]
            //   rank4  -> [B, S, IC, 1] encoded as [b=B, f=S, y=IC, x=1]
            if (x[3] != 1)
                return false;
            if (x[2] == 1) {
                mb = static_cast<int64_t>(x[0]);
                ic = static_cast<int64_t>(x[1]);
            } else {
                mb = static_cast<int64_t>(x[0]) * static_cast<int64_t>(x[1]);
                ic = static_cast<int64_t>(x[2]);
            }
        }

        // Weights supported as 2D matrices encoded as [b, f, 1, 1]
        if (w_gate[2] != 1 || w_gate[3] != 1 || w_up[2] != 1 || w_up[3] != 1 || w_down[2] != 1 || w_down[3] != 1)
            return false;

        const int64_t w_gate_ic = static_cast<int64_t>(w_gate[0]);
        const int64_t w_gate_oc = static_cast<int64_t>(w_gate[1]);
        const int64_t w_up_ic = static_cast<int64_t>(w_up[0]);
        const int64_t w_up_oc = static_cast<int64_t>(w_up[1]);
        const int64_t w_down_oc = static_cast<int64_t>(w_down[0]);
        const int64_t w_down_ic = static_cast<int64_t>(w_down[1]);

        if (w_gate_ic != ic || w_up_ic != ic)
            return false;
        if (w_up_oc != w_gate_oc)
            return false;
        if (w_down_oc != w_gate_oc || w_down_ic != ic)
            return false;

        oc = w_gate_oc;
        return true;
    }

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fused_mlp>());

        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();

        if (!info.supports_immad || info.arch == gpu_arch::unknown || !config.get_use_onednn())
            LOG_AND_RETURN_FALSE(node);

        const auto& fm_node = node.as<fused_mlp>();
        const auto& x_layout = fm_node.get_input_layout(0);
        const auto& out_layout = fm_node.get_output_layout(0);

        if (x_layout.data_type != data_types::f16 || out_layout.data_type != data_types::f16)
            LOG_AND_RETURN_FALSE(node);
        for (size_t i = 1; i < 4; ++i) {
            if (fm_node.get_input_layout(i).data_type != data_types::f16)
                LOG_AND_RETURN_FALSE(node);
        }

        if (x_layout.format.is_image_2d() || out_layout.format.is_image_2d())
            LOG_AND_RETURN_FALSE(node);

        // oneDNN Graph tensor API doesn't support cl_mem sub-buffer offset; require no padding/offset (POC).
        if (x_layout.data_padding || out_layout.data_padding)
            LOG_AND_RETURN_FALSE(node);
        for (size_t i = 1; i < 4; ++i) {
            if (fm_node.get_input_layout(i).data_padding)
                LOG_AND_RETURN_FALSE(node);
        }

        int64_t mb = 0, ic = 0, oc = 0;
        if (!try_extract_dims(fm_node, mb, ic, oc))
            LOG_AND_RETURN_FALSE(node);

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<fused_mlp>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        auto target_format = format::get_default_format(out_rank);

        for (size_t idx = 0; idx < node.get_dependencies().size(); ++idx) {
            if (node.get_dependency(idx).is_constant())
                continue;
            in_fmts[idx] = target_format;
        }
        out_fmts[0] = target_format;
        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn

