// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/runtime/layout.hpp"

#include <memory>
namespace cldnn {
namespace ocl {

struct ConvolutionImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::conv")
    ConvolutionImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<convolution>());

        const auto& input_layout = node.get_input_layout(0);
        const auto& weights_layout = node.as<convolution>().weights().get_output_layout();
        const auto& output_layout = node.get_output_layout(0);

        auto input_fmt = input_layout.format;
        auto output_fmt = output_layout.format;

        auto in_dt = input_layout.data_type;
        auto wei_dt = weights_layout.data_type;
        auto out_dt = output_layout.data_type;

        static const std::vector<ov::element::Type_t> supported_activation_types = {
            data_types::f32,
            data_types::f16,
            data_types::i8,
            data_types::u8
        };

        static const std::vector<ov::element::Type_t> supported_weights_types = {
            data_types::f32,
            data_types::f16,
            data_types::i8,
            data_types::u8,
            data_types::u4,
            data_types::i4,
        };

        if (!one_of(in_dt, supported_activation_types) ||
            !one_of(wei_dt, supported_weights_types) ||
            !one_of(out_dt, supported_activation_types))
            return false;

        if (m_shape_type == shape_types::dynamic_shape) {
            static const std::vector<format::type> supported_dyn_formats = {
                format::bfyx,
                format::bfzyx,
                format::b_fs_yx_fsv16,
                format::b_fs_zyx_fsv16
            };

            if (!one_of(input_fmt.value, supported_dyn_formats) || !one_of(output_fmt.value, supported_dyn_formats))
                return false;
        } else {
            static const std::vector<format::type> supported_fp_only_formats = {
                format::yxfb,
                format::winograd_2x3_s1_data,
                format::bs_fs_zyx_bsv16_fsv16,
            };
            static const std::vector<format::type> supported_int_only_formats = {
                format::b_fs_yx_fsv4,
                format::b_fs_zyx_fsv32,
            };
            static const std::vector<format::type> supported_common_formats = {
                format::bfyx,
                format::bfzyx,
                format::byxf,
                format::b_fs_yx_fsv16,
                format::b_fs_zyx_fsv16,
                format::b_fs_yx_fsv32,
                format::bs_fs_yx_bsv16_fsv16,
                format::bs_fs_yx_bsv32_fsv32,
                format::bs_fs_yx_bsv32_fsv16,
                format::bs_fs_yx_bsv4_fsv4,
                format::bs_fs_yx_bsv8_fsv4,
                format::bs_fs_yx_bsv4_fsv2,
            };

            bool fp_common_case = data_type_traits::is_floating_point(in_dt) &&
                           (one_of(input_fmt.value, supported_fp_only_formats) || one_of(input_fmt.value, supported_common_formats));
            bool fp16_case = everyone_is(ov::element::f16, in_dt, wei_dt) && (input_fmt == format::fs_b_yx_fsv32 || output_fmt == format::fs_b_yx_fsv32);
            bool i8u8_case = data_type_traits::is_i8_u8(in_dt) &&
                             (one_of(input_fmt.value, supported_int_only_formats) || one_of(input_fmt.value, supported_common_formats));

            if (!fp_common_case && !fp16_case && !i8u8_case)
                return false;
        }

        return true;
    }
};

}  // namespace ocl
}  // namespace cldnn
