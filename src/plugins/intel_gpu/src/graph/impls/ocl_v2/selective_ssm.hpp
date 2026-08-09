// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "selective_ssm_inst.h"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct SelectiveSSMOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::selective_ssm::opt")
    explicit SelectiveSSMOpt(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<selective_ssm>());
        static constexpr std::array supported_fmts = {format::bfyx};
        static constexpr std::array supported_types = {ov::element::f16, ov::element::f32, ov::element::bf16};

        ov::element::Type common_type = ov::element::dynamic;
        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            if (!one_of(in_layout.format, supported_fmts) || !one_of(in_layout.data_type, supported_types)) {
                return false;
            }
            if (common_type.is_dynamic()) {
                common_type = in_layout.data_type;
            } else if (common_type != in_layout.data_type) {
                return false;
            }
        }

        for (size_t i = 0; i < node.get_outputs_count(); i++) {
            const auto& out_layout = node.get_output_layout(i);
            if (!one_of(out_layout.format, supported_fmts) || !one_of(out_layout.data_type, supported_types)) {
                return false;
            }
            if (!common_type.is_dynamic() && out_layout.data_type != common_type) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
