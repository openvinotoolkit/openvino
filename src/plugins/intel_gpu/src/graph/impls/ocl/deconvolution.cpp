// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "deconvolution_inst.h"
#include "intel_gpu/plugin/common_utils.hpp"
#include "deconvolution/deconvolution_kernel_selector.h"
#include "deconvolution/deconvolution_kernel_base.h"

namespace cldnn {
namespace ocl {

struct deconvolution_impl : typed_primitive_impl_ocl<deconvolution> {
    using parent = typed_primitive_impl_ocl<deconvolution>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::deconvolution_kernel_selector;
    using kernel_params_t = kernel_selector::deconvolution_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::deconvolution_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<deconvolution_impl, kernel_params_t>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<deconvolution>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        args.weights = instance.weights_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<deconvolution>();
        const auto& stride = primitive->stride;
        const ov::Strides dilation(impl_param.get_output_layout().get_spatial_rank(), 1);

        const auto& pad = primitive->pad;
        const auto& groups = primitive->groups;

        auto params = get_weights_bias_default_params<kernel_selector::deconvolution_params>(impl_param, primitive->grouped_weights_shape);

        params.groups = groups;

        const auto weights_idx = 1 + 0;
        const auto& weights_layout = impl_param.input_layouts[weights_idx].convert_to_weights_layout(primitive->grouped_weights_shape);
        uint32_t kx = weights_layout.spatial(0);
        uint32_t ky = weights_layout.spatial(1);
        uint32_t kz = weights_layout.spatial(2);

        params.filterSize = { kx, ky, kz };

        uint32_t pad_x, pad_y, pad_z;
        std::tie(pad_x, pad_y, pad_z) = ov::intel_gpu::get_xyz<ov::CoordinateDiff, uint32_t>(pad, 0);
        params.padding = {pad_x, pad_y, pad_z};

        uint32_t stride_x, stride_y, stride_z;
        std::tie(stride_x, stride_y, stride_z) = ov::intel_gpu::get_xyz<ov::Strides, uint32_t>(stride, 1);
        params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_x, dilation_y, dilation_z;
        std::tie(dilation_x, dilation_y, dilation_z) = ov::intel_gpu::get_xyz<ov::Strides, uint32_t>(dilation, 1);
        params.dilation = {dilation_x, dilation_y, dilation_z};

        return params;
    }
};

namespace detail {

attach_deconvolution_impl::attach_deconvolution_impl() {
    static auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8};
    static auto formats = {
        format::bfyx,
        format::byxf,
        format::yxfb,

        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
    };

    implementation_map<deconvolution>::add(impl_types::ocl, typed_primitive_impl_ocl<deconvolution>::create<deconvolution_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::deconvolution_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::deconvolution)
