// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "intel_gpu/runtime/format.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/utils.hpp"

#include "registry/implementation_manager.hpp"

#include "utils.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct ConvolutionImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::conv")
    ConvolutionImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<convolution>());
        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown || !config.get_use_onednn())
            return false;

        const auto& conv_node = node.as<convolution>();

        const auto& in_layout = conv_node.get_input_layout(0);
        const auto& out_layout = conv_node.get_output_layout(0);
        const auto& wei_layout = conv_node.weights().get_output_layout(false);

        auto in_fmt = in_layout.format;
        auto out_fmt = out_layout.format;

        auto in_dt = in_layout.data_type;
        auto wei_dt = wei_layout.data_type;
        auto out_dt = out_layout.data_type;

        static const std::vector<format> supported_formats = {
            format::any,
            format::bfyx,
            format::bfzyx,
            format::byxf,
            format::bzyxf,
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

        if (!one_of(in_fmt, supported_formats) || !one_of(out_fmt, supported_formats))
            return false;

        auto prim = conv_node.get_primitive();
        if (prim->groups > 1 && !prim->grouped_weights_shape)
            return false;

        if (!is_supported_pad(in_layout) || !is_supported_pad(out_layout))
            return false;

        bool f16_conv = everyone_is(data_types::f16, in_dt, wei_dt) && one_of(out_dt, {data_types::f16, data_types::f32, data_types::u8, data_types::i8});
        bool u8s8_conv = one_of(in_dt, {data_types::i8, data_types::u8}) &&
                         wei_dt == data_types::i8 &&
                         one_of(out_dt, {data_types::i32, data_types::f16, data_types::f32, data_types::u8, data_types::i8});

        if (!f16_conv && !u8s8_conv)
            return false;

        if (!is_supported_post_ops(conv_node))
            return false;

        if (prim->deformable_mode)
            return false;

        // oneDNN only supports asymmetric weights quantization by scalar zero-points
        if (conv_node.weights_zero_points_term() &&
            conv_node.weights_zero_points().get_output_layout().count() != 1)
            return false;

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override;
};

}  // namespace onednn
}  // namespace cldnn
