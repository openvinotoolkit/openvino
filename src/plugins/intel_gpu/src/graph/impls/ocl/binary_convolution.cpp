// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/quantize.hpp"
#include "binary_convolution_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "kernel_selector/core/actual_kernels/binary_convolution/binary_convolution_kernel_selector.h"
#include "kernel_selector/core/actual_kernels/binary_convolution/binary_convolution_params.h"
#include <algorithm>
#include <memory>

namespace cldnn {
namespace ocl {

struct binary_convolution_impl : typed_primitive_impl_ocl<binary_convolution> {
    using parent = typed_primitive_impl_ocl<binary_convolution>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<binary_convolution_impl>(*this);
    }

protected:
    bool validate_impl(const typed_primitive_inst<binary_convolution>& instance) const override {
        bool res = true;

        auto outer_id = _outer.id();
        auto data_type = instance.node.input().get_output_layout().data_type;

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        CLDNN_ERROR_DATA_TYPES_MISMATCH(outer_id,
                                        "Input memory",
                                        data_type,
                                        "output memory",
                                        instance.node.get_output_layout().data_type,
                                        "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(outer_id,
                                                    "Input memory",
                                                    data_type,
                                                    "filter memory",
                                                    instance.weights_memory(0)->get_layout().data_type,
                                                    "");

        return res;
    }

    kernel_arguments_data get_arguments(typed_primitive_inst<binary_convolution>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = instance.weights_memory(split);
        return args;
    }

    int32_t get_split() const override { return _outer.get_split(); }

public:
    static primitive_impl* create(const binary_convolution_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& weights_layout = arg.weights(0).get_output_layout().convert_to_weights_layout(false);
        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& groups = primitive->groups;
        const auto& stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& pad = primitive->pad;

        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        const auto actual_split = depthwise_separable_opt ? (decltype(split))1 : split;

        assert(arg.get_output_layout().feature() / primitive->split() == weights_layout.batch());

        auto conv_params =
            get_weights_bias_default_params<kernel_selector::binary_convolution_params>(arg, actual_split);
        auto conv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::binary_convolution_optional_params>(
                arg.get_program());

        conv_params.pad_value = primitive->pad_value;
        conv_params.out_dt = to_data_type(*primitive->output_data_type);
        conv_params.depthwise_separable_opt = depthwise_separable_opt;
        conv_params.split = static_cast<uint32_t>(split);
        conv_params.groups = static_cast<uint32_t>(groups);
        conv_params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
            (uint32_t)weights_size.spatial[2],
        };

        uint32_t pad_z = std::max<std::ptrdiff_t>(pad.size() >= 3 ? pad[pad.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pad.size() >= 2 ? pad[pad.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pad.size() >= 1 ? pad[pad.size() - 1] : 0, 0);
        conv_params.padding = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
        uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
        uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
        conv_params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;
        conv_params.dilation = {dilation_x, dilation_y, dilation_z};

        auto& kernel_selector = kernel_selector::binary_convolution_kernel_selector::Instance();

        const auto& tuning_config = arg.get_program().get_options().get<build_option_type::tuning_config>();

        if (tuning_config->config.mode == tuning_mode::tuning_tune_and_cache ||
            tuning_config->config.mode == tuning_mode::tuning_retune_and_cache) {
            conv_optional_params.tuningParams.runner =
                std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), arg.get_program().get_id(), true);
        }

        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto conv = new binary_convolution_impl(arg, best_kernels[0]);

        return conv;
    }
};

namespace detail {

attach_binary_convolution_impl::attach_binary_convolution_impl() {
    implementation_map<binary_convolution>::add(impl_types::ocl, binary_convolution_impl::create, {
        std::make_tuple(data_types::bin, format::b_fs_yx_32fp),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
