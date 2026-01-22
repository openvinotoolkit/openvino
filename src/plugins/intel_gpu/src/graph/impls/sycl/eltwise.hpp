// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_inst.h"
#include "registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace sycl {

struct EltwiseImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("EltwiseImplementationManager")
    EltwiseImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::sycl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<eltwise>());

        static const std::vector<format::type> supported_formats = {
            format::any,
            format::bfyx,
        };

        static const std::vector<ov::element::Type_t> supported_types = {
            ov::element::f32,
            ov::element::f16,
        };

        const auto& eltwise_node = node.as<eltwise>();
        const auto& in0_layout = eltwise_node.get_input_layout(0);
        const auto& in1_layout = eltwise_node.get_input_layout(1);
        const auto& out_layout = eltwise_node.get_output_layout(0);
        auto in0_dt = in0_layout.data_type;
        auto in1_dt = in1_layout.data_type;
        auto out_dt = out_layout.data_type;

        if (!one_of(in0_dt, supported_types) || !one_of(in1_dt, supported_types) || !one_of(out_dt, supported_types))
            return false;

        if (!one_of(in0_layout.format.value, supported_formats) || !one_of(in1_layout.format.value, supported_formats) ||
            !one_of(out_layout.format.value, supported_formats))
            return false;

        if (in0_layout.data_padding || in1_layout.data_padding || out_layout.data_padding)
            return false;

        return true;
    }
};

}  // namespace sycl
}  // namespace cldnn
