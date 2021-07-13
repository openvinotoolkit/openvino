// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
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
        const auto& weights_layout = arg.weights(0).get_output_layout();

        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& stride = primitive->stride;
#if 0  // TODO: support dilation
        const auto& dilation = primitive->dilation;
#else
        const tensor dilation = {0, 0, 1, 1, 1};
#endif
        const auto actual_split = split;

        const auto& input_offset = primitive->input_offset;
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

        auto spatial_size = arg.get_output_layout().format.dimension() - 2;
        uint32_t kx = weights_size.spatial[0];
        uint32_t ky = weights_size.spatial[1];
        uint32_t kz = spatial_size == 2 ? 1 : weights_size.spatial[2];
        deconv_params.filterSize = { kx, ky, kz };

        deconv_params.padding = {(uint32_t)std::max(-input_offset.spatial[0], 0),
                                 (uint32_t)std::max(-input_offset.spatial[1], 0),
                                 (uint32_t)std::max(-input_offset.spatial[2], 0)};

        deconv_params.stride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1], (uint32_t)stride.spatial[2]};

        deconv_params.dilation = {(uint32_t)dilation.spatial[0],
                                  (uint32_t)dilation.spatial[1],
                                  (uint32_t)dilation.spatial[2]};

        auto& kernel_selector = kernel_selector::deconvolution_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(deconv_params, deconv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");
        auto deconv = new deconvolution_impl(arg, best_kernels[0]);

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
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
