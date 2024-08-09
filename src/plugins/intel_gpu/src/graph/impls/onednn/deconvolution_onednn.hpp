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
    OV_GPU_PRIMITIVE_IMPL("DeconvolutionImplementationOnednn")
    DeconvolutionImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<deconvolution>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad)
            return false;
        const auto& deconv_node = node.as<deconvolution>();
        static const std::vector<format::type> supported_formats = {
            format::bfyx,
            format::byxf,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv32,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bs_fs_yx_bsv4_fsv4,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_yx_bsv4_fsv2,
        };

        if (!one_of(node.get_preferred_input_fmt(0), supported_formats))
            return false;

        const auto& input_layout = deconv_node.get_input_layout(0);
        auto in_dt = input_layout.data_type;
        auto wei_dt = deconv_node.weights().get_output_layout().data_type;
        auto out_dt = deconv_node.get_output_layout(false).data_type;

        const auto& prim = deconv_node.get_primitive();

        if (prim->groups != 1)
            return false;

        auto spatial_dims_num = input_layout.get_spatial_rank();

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

        return ImplementationManager::validate(node);
    }

    in_out_fmts_t query_formats(const program_node& node) const override;

    bool support_shapes(const kernel_impl_params& params) const override {
        return true;
    }
};

}  // namespace onednn
}  // namespace cldnn
