// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry/implementation_manager.hpp"
#include "program_node.h"

namespace cldnn {
namespace ocl {

struct LoraImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("lora")

    LoraImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
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

}  // namespace ocl
}  // namespace cldnn
