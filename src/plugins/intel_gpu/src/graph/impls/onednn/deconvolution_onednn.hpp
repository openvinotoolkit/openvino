// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_inst.h"
#include "impls/onednn/utils.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "impls/registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct DeconvolutionImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::deconv")
    DeconvolutionImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<deconvolution>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown)
            return false;

        const auto& deconv_node = node.as<deconvolution>();
        static const std::vector<format::type> supported_formats = {
            format::any,
            format::bfyx,
            format::bfzyx,
            format::byxf,
            format::b_fs_yx_fsv8,
            format::b_fs_zyx_fsv8,
            format::b_fs_yx_fsv16,
            format::b_fs_zyx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv32,
            format::bs_fs_yx_bsv4_fsv2,
            format::bs_fs_yx_bsv4_fsv4,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_zyx_bsv8_fsv2,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_zyx_bsv8_fsv4,
            format::bs_fs_yx_bsv16_fsv2,
            format::bs_fs_zyx_bsv16_fsv2,
            format::bs_fs_yx_bsv16_fsv4,
            format::bs_fs_zyx_bsv16_fsv4,
            format::bs_fs_yx_bsv16_fsv8,
            format::bs_fs_zyx_bsv16_fsv8,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bs_fs_zyx_bsv32_fsv32,
        };


        const auto& input_layout = deconv_node.get_input_layout(0);
        const auto& output_layout = deconv_node.get_output_layout(0);

        auto in_fmt = input_layout.format;
        auto out_fmt = output_layout.format;

        auto in_dt = input_layout.data_type;
        auto wei_dt = deconv_node.weights().get_output_layout(false).data_type;
        auto out_dt = output_layout.data_type;

        if (!is_supported_pad(input_layout) || !is_supported_pad(output_layout))
            return false;

        if (!one_of(in_fmt.value, supported_formats) || !one_of(out_fmt.value, supported_formats))
            return false;

        const auto& prim = deconv_node.get_primitive();

        if (prim->groups != 1)
            return false;

        auto spatial_dims_num = input_layout.get_partial_shape().size() - 2;

        if (spatial_dims_num > 3)
            return false;

        bool f16_deconv = everyone_is(data_types::f16, in_dt, wei_dt) && one_of(out_dt, {data_types::f16, data_types::u8, data_types::i8});
        bool f32_deconv = everyone_is(data_types::f32, in_dt, wei_dt) && one_of(out_dt, {data_types::u8, data_types::i8});
        bool u8s8_deconv = one_of(in_dt, {data_types::i8, data_types::u8}) &&
                           wei_dt == data_types::i8 &&
                           one_of(out_dt, {data_types::i32, data_types::f16, data_types::f32, data_types::u8, data_types::i8});

        if (!f16_deconv && !f32_deconv && !u8s8_deconv)
            return false;

        if (!is_supported_post_ops(deconv_node))
            return false;

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override;
};

}  // namespace onednn
}  // namespace cldnn
