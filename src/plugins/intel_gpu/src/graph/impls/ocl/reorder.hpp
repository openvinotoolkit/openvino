// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/implementation_manager.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "program_node.h"

#include <memory>
namespace cldnn {
namespace ocl {

struct ReorderImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ReorderImplementationOCL")
    ReorderImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::ocl, shape_type) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    std::unique_ptr<primitive_impl> create_impl(const kernel_impl_params& params) const override;

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<reorder>());
        if (!node.is_dynamic())
            return true;

        static const std::vector<format::type> supported_dyn_formats = {
            format::bfyx,
            format::bfzyx,
            format::bfwzyx,
        };
        static const std::vector<ov::element::Type_t> supported_dyn_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::u8,
            ov::element::i8,
            ov::element::i32,
            ov::element::i64,
        };

        const auto& input_layout = node.get_input_layout(0);
        const auto& output_layout = node.get_output_layout(0);

        auto input_fmt = input_layout.format;
        auto output_fmt = output_layout.format;

        auto in_dt = input_layout.data_type;
        auto out_dt = output_layout.data_type;

        if (output_fmt == format::custom)
            return false;

        if (m_shape_type == shape_types::dynamic_shape) {
            if (!one_of(input_fmt.value, supported_dyn_formats) || !one_of(output_fmt.value, supported_dyn_formats))
                return false;

            if (!one_of(in_dt, supported_dyn_types) || !one_of(out_dt, supported_dyn_types))
                return false;
        }

        return true;
    }

    in_out_fmts_t query_formats(const program_node&) const override { OPENVINO_NOT_IMPLEMENTED; }
    bool support_shapes(const kernel_impl_params&) const override { return true; }
};

}  // namespace ocl
}  // namespace cldnn
