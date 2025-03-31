// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "col2im_inst.h"
#include "col2im/col2im_kernel_selector.h"
#include "col2im/col2im_kernel_ref.h"

#include "intel_gpu/plugin/common_utils.hpp"

namespace cldnn {
namespace ocl {
struct col2im_impl : typed_primitive_impl_ocl<col2im> {
    using parent = typed_primitive_impl_ocl<col2im>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::col2im_kernel_selector;
    using kernel_params_t = kernel_selector::col2im_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::col2im_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<col2im_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<col2im>();
        auto col2im_params = get_default_params<kernel_selector::col2im_params>(impl_param);

        // Attributes
        uint32_t stride_x, stride_y, stride_z;
        uint32_t dilation_x, dilation_y, dilation_z;
        std::tie(stride_x, stride_y, stride_z) = ov::intel_gpu::get_xyz<ov::Strides, uint32_t>(primitive->stride, 1);
        col2im_params.stride = {stride_x, stride_y, stride_z};
        std::tie(dilation_x, dilation_y, dilation_z) = ov::intel_gpu::get_xyz<ov::Strides, uint32_t>(primitive->dilation, 1);
        col2im_params.dilation = {dilation_x, dilation_y, dilation_z};

        // padding being & end
        uint32_t pad_begin_x, pad_begin_y, pad_begin_z;
        std::tie(pad_begin_x, pad_begin_y, pad_begin_z) = ov::intel_gpu::get_xyz<ov::CoordinateDiff, uint32_t>(primitive->padding_begin, 0);
        col2im_params.padding_begin = {pad_begin_x, pad_begin_y, pad_begin_z};
        uint32_t pad_end_x, pad_end_y, pad_end_z;
        std::tie(pad_end_x, pad_end_y, pad_end_z) = ov::intel_gpu::get_xyz<ov::CoordinateDiff, uint32_t>(primitive->padding_end, 0);
        col2im_params.padding_end = {pad_end_x, pad_end_y, pad_end_z};

        // Col2Im-15 implementation : required
        // Output size is 1D tensor of two positive integer numbers (height and width). Kernel size is non-negative integer numbers.
        std::vector<size_t> output_size((primitive->output_shape.begin()), primitive->output_shape.end());
        std::vector<size_t> kernel_size(primitive->kernel_shape.begin(), primitive->kernel_shape.end());
        col2im_params.output_size = {(uint32_t)output_size[0], (uint32_t)output_size[1], (uint32_t)1};
        col2im_params.kernel_size = {(uint32_t)kernel_size[0], (uint32_t)kernel_size[1], (uint32_t)1};

        return col2im_params;
    }
};

namespace detail {

attach_col2im_impl::attach_col2im_impl() {
    std::vector<data_types> dt = {
        data_types::f16,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
    };
    implementation_map<col2im>::add(impl_types::ocl, typed_primitive_impl_ocl<col2im>::create<col2im_impl>, dt, fmt);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::col2im_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::col2im)
