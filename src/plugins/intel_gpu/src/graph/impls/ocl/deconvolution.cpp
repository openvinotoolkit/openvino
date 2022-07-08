// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "deconvolution/deconvolution_kernel_selector.h"
#include "deconvolution/deconvolution_kernel_base.h"
#include <algorithm>

namespace cldnn {
namespace ocl {

struct deconvolution_impl : typed_primitive_impl_ocl<deconvolution> {
    using parent = typed_primitive_impl_ocl<deconvolution>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deconvolution_impl>(*this);
    }

protected:
    // TODO: share it with convolution and fully connected
    bool validate_impl(const typed_primitive_inst<deconvolution>&) const override {
        bool res = true;

        CLDNN_ERROR_NOT_EQUAL(_outer.id(),
                              "deconvolution filling value",
                              _outer.get_output_layout().data_padding.filling_value(),
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in deconvolution.");

        return res;
    }

    kernel_arguments_data get_arguments(typed_primitive_inst<deconvolution>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = instance.weights_memory(split);
        args.bias = instance.bias_term() ? instance.bias_memory(split) : nullptr;

        return args;
    }

    int32_t get_split() const override { return _outer.get_split(); }

    uint32_t get_groups() const override { return _outer.get_groups(); }

public:
    static primitive_impl* create(const deconvolution_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& weights_layout = arg.weights(0).get_output_layout().convert_to_weights_layout(primitive->grouped_weights_shape);

        const auto& split = primitive->split();
        const auto& stride = primitive->stride;
#if 0  // TODO: support dilation
        const auto& dilation = primitive->dilation;
#else
        const ov::Strides dilation(arg.get_output_layout().get_spatial_rank(), 1);
#endif
        const auto actual_split = split;

        const auto& pad = primitive->pad;
        const auto& groups = primitive->groups;

        auto deconv_params = get_weights_bias_default_params<kernel_selector::deconvolution_params>(
            arg,
            (groups > 1) ? 1 : actual_split,
            1,
            primitive->grouped_weights_shape);
        auto deconv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::deconvolution_optional_params>(arg.get_program());

        deconv_params.split = split;
        deconv_params.groups = groups;

        uint32_t kx = weights_layout.spatial(0);
        uint32_t ky = weights_layout.spatial(1);
        uint32_t kz = weights_layout.spatial(2);

        deconv_params.filterSize = { kx, ky, kz };

        uint32_t pad_z = std::max<std::ptrdiff_t>(pad.size() >= 3 ? pad[pad.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pad.size() >= 2 ? pad[pad.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pad.size() >= 1 ? pad[pad.size() - 1] : 0, 0);
        deconv_params.padding = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
        uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
        uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
        deconv_params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;
        deconv_params.dilation = {dilation_x, dilation_y, dilation_z};

        auto& kernel_selector = kernel_selector::deconvolution_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(deconv_params, deconv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");
        auto deconv = new deconvolution_impl(arg, best_kernels[0]);
        deconv->can_reuse_memory = best_kernels[0].can_reuse_memory;

        return deconv;
    }
};

namespace detail {

attach_deconvolution_impl::attach_deconvolution_impl() {
    implementation_map<deconvolution>::add(impl_types::ocl, deconvolution_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
