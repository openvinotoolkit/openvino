// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deformable_convolution_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "convolution/convolution_kernel_selector.h"
#include "convolution/convolution_params.h"
#include <algorithm>

namespace cldnn {
namespace ocl {

struct deformable_conv_impl : typed_primitive_impl_ocl<deformable_conv> {
    using parent = typed_primitive_impl_ocl<deformable_conv>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deformable_conv_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<deformable_conv>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = instance.weights_memory(split);
        args.bias = instance.bias_term() ? instance.bias_memory(split) : nullptr;
        return args;
    }

    int32_t get_split() const override { return _outer.get_split(); }

    uint32_t get_groups() const override { return _outer.get_groups(); }

public:
    static primitive_impl* create(const deformable_conv_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& weights_layout = arg.weights(0).get_output_layout();
        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& groups = primitive->groups;

        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        const auto actual_split = depthwise_separable_opt ? (decltype(split))1 : split;

        auto conv_params = get_weights_bias_default_params<kernel_selector::convolution_params>(
            arg,
            (groups > 1 && !depthwise_separable_opt) ? groups : actual_split,
            groups);
        auto conv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::convolution_optional_params>(arg.get_program());

        conv_params.depthwise_separable_opt = depthwise_separable_opt;
        conv_params.split = split;
        conv_params.groups = groups;
        conv_params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
            (uint32_t)weights_size.spatial[2],
        };

        auto& kernel_selector = kernel_selector::deformable_conv_kernel_selector::Instance();
        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");
        auto conv = new deformable_conv_impl(arg, best_kernels[0]);

        return conv;
    }
};

struct deformable_interp_impl : typed_primitive_impl_ocl<deformable_interp> {
    using parent = typed_primitive_impl_ocl<deformable_interp>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deformable_interp_impl>(*this);
    }

protected:
    int32_t get_split() const override { return 1; }

    uint32_t get_groups() const override { return 1; }

public:
    static primitive_impl* create(const deformable_interp_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& input_layout = arg.input().get_output_layout();
        const auto& kernel_size = primitive->kernel_size;

        const auto& stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& input_offset = primitive->input_offset;
        const auto& groups = primitive->groups;
        const auto& deformable_groups = primitive->deformable_groups;

        auto conv_params = get_default_params<kernel_selector::convolution_params>(arg, groups);
        auto conv_optional_params =
            get_default_optional_params<kernel_selector::convolution_optional_params>(arg.get_program());

        // It's not really needed, just initialize fields of params
        auto weights_layout = layout(input_layout.data_type, input_layout.format, kernel_size);
        conv_params.weights = convert_weights_tensor(weights_layout);

        const auto additional_offset = tensor::max(input_offset, (tensor) 0);
        if (additional_offset != (tensor) 0) {
            conv_params.inputs[0] = convert_data_tensor(input_layout, groups, additional_offset);
        }

        conv_params.inputs.push_back(convert_data_tensor(arg.trans().get_output_layout()));
        conv_params.deformable_groups = deformable_groups;

        conv_params.padding = {(uint32_t)std::max(-input_offset.spatial[0], 0),
                               (uint32_t)std::max(-input_offset.spatial[1], 0),
                               (uint32_t)std::max(-input_offset.spatial[2], 0)};

        conv_params.stride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1], (uint32_t)stride.spatial[2]};

        conv_params.kernelSize = { (uint32_t)kernel_size.spatial[0],
                                   (uint32_t)kernel_size.spatial[1],
                                   (uint32_t)kernel_size.spatial[2] };

        conv_params.dilation = {(uint32_t)dilation.spatial[0],
                                (uint32_t)dilation.spatial[1],
                                (uint32_t)dilation.spatial[2]};

        auto& kernel_selector = kernel_selector::deformable_interp_kernel_selector::Instance();
        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");
        auto conv = new deformable_interp_impl(arg, best_kernels[0]);

        return conv;
    }
};

namespace detail {

attach_deformable_conv_impl::attach_deformable_conv_impl() {
    implementation_map<deformable_conv>::add(impl_types::ocl, deformable_conv_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

attach_deformable_interp_impl::attach_deformable_interp_impl() {
    implementation_map<deformable_interp>::add(impl_types::ocl, deformable_interp_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
