// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "deformable_convolution_inst.h"
#include "convolution/convolution_kernel_selector.h"
#include "convolution/convolution_params.h"

namespace cldnn {
namespace ocl {

struct deformable_conv_impl : typed_primitive_impl_ocl<deformable_conv> {
    using parent = typed_primitive_impl_ocl<deformable_conv>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::deformable_conv_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::convolution_params, kernel_selector::convolution_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deformable_conv_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<deformable_conv>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        args.weights = instance.weights_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<deformable_conv>();
        const auto& groups = primitive->groups;

        auto params = get_weights_bias_default_params<kernel_selector::convolution_params>(impl_param);
        auto optional_params = get_default_weights_bias_optional_params<kernel_selector::convolution_optional_params>(impl_param.get_program());

        const auto weight_idx = 1 + 0;
        const auto& weights_layout = impl_param.input_layouts[weight_idx].convert_to_weights_layout(false);
        const auto& weights_size = weights_layout.get_tensor();

        params.groups = groups;
        params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
            (uint32_t)weights_size.spatial[2],
        };

        return {params, optional_params};
    }
};

struct deformable_interp_impl : typed_primitive_impl_ocl<deformable_interp> {
    using parent = typed_primitive_impl_ocl<deformable_interp>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::deformable_interp_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::convolution_params, kernel_selector::convolution_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deformable_interp_impl>(*this);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<deformable_interp>();
        const auto input_idx = 0;
        const auto trans_idx = 1;
        const auto mask_idx = 2;
        const auto& input_layout = impl_param.input_layouts[input_idx];
        const auto& kernel_size = primitive->kernel_size;

        auto stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& pad = primitive->pad;
        const auto& deformable_groups = primitive->deformable_groups;

        auto params = get_default_params<kernel_selector::convolution_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::convolution_optional_params>(impl_param.get_program());

        // It's not really needed, just initialize fields of params
        auto weights_layout = layout(input_layout.data_type, input_layout.format, kernel_size);
        params.weights = convert_weights_tensor(weights_layout);

        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[trans_idx]));
        if (primitive->input.size() == 3) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[mask_idx]));
            params.deformable_mask_enabled = true;
        }
        params.bilinear_interpolation_pad = primitive->bilinear_interpolation_pad;
        params.deformable_groups = deformable_groups;

        uint32_t pad_z = std::max<std::ptrdiff_t>(pad.size() >= 3 ? pad[pad.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pad.size() >= 2 ? pad[pad.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pad.size() >= 1 ? pad[pad.size() - 1] : 0, 0);

        params.padding = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
        uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
        uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
        params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;
        params.dilation = {dilation_x, dilation_y, dilation_z};

        params.kernelSize = { (uint32_t)kernel_size.spatial[0],
                              (uint32_t)kernel_size.spatial[1],
                              (uint32_t)kernel_size.spatial[2] };
        return {params, optional_params};
    }
};

namespace detail {

attach_deformable_conv_impl::attach_deformable_conv_impl() {
    implementation_map<deformable_conv>::add(impl_types::ocl, typed_primitive_impl_ocl<deformable_conv>::create<deformable_conv_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

attach_deformable_interp_impl::attach_deformable_interp_impl() {
    implementation_map<deformable_interp>::add(impl_types::ocl, typed_primitive_impl_ocl<deformable_interp>::create<deformable_interp_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::deformable_conv_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::deformable_interp_impl)
