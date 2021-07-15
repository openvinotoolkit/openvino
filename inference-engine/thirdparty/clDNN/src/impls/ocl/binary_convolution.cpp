// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/primitives/scale.hpp"
#include "cldnn/primitives/quantize.hpp"
#include "binary_convolution_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
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
        const auto& input_layout = arg.input().get_output_layout();
        const auto& weights_layout = arg.weights(0).get_output_layout();
        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& groups = primitive->groups;
        const auto& stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& input_offset = primitive->input_offset;

        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        const auto actual_split = depthwise_separable_opt ? (decltype(split))1 : split;

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        auto conv_params =
            get_weights_bias_default_params<kernel_selector::binary_convolution_params>(arg, actual_split);
        auto conv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::binary_convolution_optional_params>(
                arg.get_program());

        const auto additional_offset = tensor::max(input_offset, (tensor) 0);
        if (additional_offset != (tensor) 0) {
            conv_params.inputs[0] = convert_data_tensor(input_layout, actual_split, additional_offset);
        }

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

        conv_params.padding = {(uint32_t)std::max(-input_offset.spatial[0], 0),
                               (uint32_t)std::max(-input_offset.spatial[1], 0),
                               (uint32_t)std::max(-input_offset.spatial[2], 0)};

        conv_params.stride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1], (uint32_t)stride.spatial[2]};
        conv_params.dilation = {(uint32_t)dilation.spatial[0],
                                (uint32_t)dilation.spatial[1],
                                (uint32_t)dilation.spatial[2]};

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
