// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/runtime/layout.hpp"
#include "registry/implementation_manager.hpp"
#include "vl_sdpa_inst.h"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

struct VLSDPAOptImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::vl_sdpa::opt")
    explicit VLSDPAOptImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<vl_sdpa>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
            in_fmts[idx] = format::bfyx;
        }
        out_fmts[0] = format::ybfx;
        for (size_t idx = 1; idx < node.get_outputs_count(); idx++) {
            out_fmts[idx] = format::bfyx;
        }

        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<vl_sdpa>());
        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        const auto& info = engine.get_device_info();

        // CM optimized for systolic-array architectures
        if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
            return false;
        }

        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_types = {
            ov::element::f16,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::cm
