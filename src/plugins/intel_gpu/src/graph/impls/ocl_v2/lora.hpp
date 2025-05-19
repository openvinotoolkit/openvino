// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct Lora : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::lora")

    explicit Lora(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static const std::vector<format> supported_fmts = {
            format::bfyx,
        };

        static const std::vector<ov::element::Type_t> supported_types = {
            ov::element::f32,
            ov::element::f16,
        };

        for (const auto& input_layout : node.get_input_layouts()) {
            if (!one_of(input_layout.format, supported_fmts) || !one_of(input_layout.data_type, supported_types)) {
                return false;
            }
        }

        const auto& output_layout = node.get_output_layout(0);
        if (!one_of(output_layout.format, supported_fmts) || !one_of(output_layout.data_type, supported_types)) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
