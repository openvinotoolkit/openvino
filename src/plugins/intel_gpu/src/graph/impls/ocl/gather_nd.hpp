// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/implementation_manager.hpp"
#include "program_node.h"

#include <memory>
namespace cldnn {
namespace ocl {

struct GatherNDImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::gather_nd")
    GatherNDImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override {
        static const std::vector<format> supported_fmts = {
            format::bfyx,
            format::bfzyx,
            format::bfwzyx
        };

        static const std::vector<ov::element::Type_t> supported_in_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32
        };

        static const std::vector<ov::element::Type_t> supported_out_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32,
            ov::element::i8,
            ov::element::u8,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& in1_layout = node.get_input_layout(1);
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts))
            return false;

        if (!one_of(in0_layout.data_type, supported_in_types) || !one_of(in1_layout.data_type, supported_in_types))
            return false;

        if (!one_of(out_layout.data_type, supported_out_types))
            return false;

        return true;
    }
};

}  // namespace ocl
}  // namespace cldnn
