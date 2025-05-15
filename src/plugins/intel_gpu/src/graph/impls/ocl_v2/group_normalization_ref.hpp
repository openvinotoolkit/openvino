// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct GroupNormalizationRef : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::group_norm::ref")
    explicit GroupNormalizationRef(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {format::bfyx, format::bfzyx, format::b_fs_yx_fsv16};

        static constexpr std::array supported_types = {
            ov::element::f32,
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

        if (!fused_ops_are_one_of<eltwise, activation>(node.get_fused_primitives())) {
            return false;
        }

        return true;
    }
    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t i = 1; i < node.get_dependencies().size(); i++) {
            size_t in_rank = node.get_input_layout(i).get_rank();
            in_fmts[i] = format::get_default_format(in_rank);
        }

        return {in_fmts, out_fmts};
    }
};

}  // namespace ov::intel_gpu::ocl
