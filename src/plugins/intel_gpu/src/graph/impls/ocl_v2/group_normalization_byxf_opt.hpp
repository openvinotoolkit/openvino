// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "group_normalization_base.hpp"
#include "group_normalization_inst.h"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct GroupNormalizationByxfOpt : public GroupNormalizationBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::group_norm::byxf_opt")
    explicit GroupNormalizationByxfOpt(shape_types shape_type, ValidateFunc vf = nullptr) : GroupNormalizationBase(shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        if (in0_layout.format != format::byxf || out_layout.format != format::byxf || in0_layout.get_partial_shape().size() != 4) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        // The vector width and the stage-2 SLM size are compile-time functions of the feature count.
        const auto& features = in0_layout.get_partial_shape()[1];
        if (features.is_dynamic() || out_layout.get_partial_shape()[1].is_dynamic()) {
            return false;
        }

        if (!fused_ops_are_one_of<eltwise, activation>(node.get_fused_primitives())) {
            return false;
        }

        const auto group_size = static_cast<uint64_t>(features.get_length() / node.as<group_normalization>().get_primitive()->num_groups);
        return group_size * sizeof(float) <= node.get_program().get_engine().get_device_info().max_local_mem_size / 2;
    }
};

}  // namespace ov::intel_gpu::ocl
