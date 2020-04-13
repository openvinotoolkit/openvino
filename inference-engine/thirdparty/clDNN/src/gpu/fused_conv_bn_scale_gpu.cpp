/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "fused_conv_bn_scale_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "fused_conv_bn_scale/fused_conv_bn_scale_kernel_selector.h"
#include "fused_conv_bn_scale/fused_conv_bn_scale_kernel_base.h"
#include <algorithm>
#include <memory>

namespace cldnn {
namespace gpu {

struct fused_conv_bn_scale_gpu : typed_primitive_gpu_impl<fused_conv_bn_scale> {
    using parent = typed_primitive_gpu_impl<fused_conv_bn_scale>;
    using parent::parent;

protected:
    bool validate_impl(const typed_primitive_inst<fused_conv_bn_scale>& instance) const override {
        bool res = true;

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(),
                                        "Input memory",
                                        instance.node.input().get_output_layout().data_type,
                                        "output memory",
                                        instance.node.get_output_layout().data_type,
                                        "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(),
                                        "Input memory",
                                        instance.node.input().get_output_layout().data_type,
                                        "filter memory",
                                        instance.weights_memory(0).get_layout().data_type,
                                        "");

        return res;
    }

    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<fused_conv_bn_scale>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);
        auto desc = std::static_pointer_cast<const fused_conv_bn_scale>(instance.desc());

        args.weights = (memory_impl::cptr) &instance.weights_memory(split);
        args.bias = (memory_impl::cptr) (instance.bias_term() ? &instance.bias_memory(split) : nullptr);

        if (!desc->scale_bias.empty()) {
            if (instance.is_fused_in_training()) {
                args.inputs.push_back((memory_impl::cptr) (&instance.dep_memory(instance.dependencies().size() - 4)));
                args.inputs.push_back((memory_impl::cptr) (&instance.dep_memory(instance.dependencies().size() - 3)));
                args.inputs.push_back((memory_impl::cptr) (&instance.dep_memory(instance.dependencies().size() - 2)));
                args.inputs.push_back((memory_impl::cptr) (&instance.dep_memory(instance.dependencies().size() - 1)));
            } else {
                args.inputs.push_back((memory_impl::cptr) (&instance.dep_memory(instance.dependencies().size() - 1)));
            }
        } else if (instance.is_fused_in_training()) {
            args.inputs.push_back((memory_impl::cptr) (&instance.dep_memory(instance.dependencies().size() - 3)));
            args.inputs.push_back((memory_impl::cptr) (&instance.dep_memory(instance.dependencies().size() - 2)));
            args.inputs.push_back((memory_impl::cptr) (&instance.dep_memory(instance.dependencies().size() - 1)));
        }

        return args;
    }

    int32_t get_split() const override { return _outer.get_split(); }

public:
    static primitive_impl* create(const fused_conv_bn_scale_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& input_layout = arg.input().get_output_layout();
        const auto& weights_layout = arg.weights(0).get_output_layout();
        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& stride = primitive->stride;
        const auto& input_offset = primitive->input_offset;
        const auto& dilation = primitive->dilation;

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        auto fuse_params = get_weights_bias_default_params<kernel_selector::fused_conv_bn_scale_params>(arg, split);
        auto fuse_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::fused_conv_bn_scale_optional_params>(
                arg.get_program());

        const auto additional_offset = tensor::max(input_offset, (tensor) 0);
        if (additional_offset != (tensor) 0) {
            fuse_params.inputs[0] = convert_data_tensor(input_layout, split, additional_offset);
        }

        fuse_params.epsilon = arg.get_primitive()->epsilon;

        fuse_params.fused_in_training = arg.is_fused_in_training();
        fuse_params.scale_bias = arg.scale_bias_term();

        fuse_params.split = split;
        fuse_params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
        };

        fuse_params.padding = {(uint32_t)std::max(-input_offset.spatial[0], 0),
                               (uint32_t)std::max(-input_offset.spatial[1], 0)};

        fuse_params.stride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1]};

        fuse_params.dilation = {(uint32_t)dilation.spatial[0], (uint32_t)dilation.spatial[1]};

        auto& kernel_selector = kernel_selector::fused_conv_bn_scale_kernel_selector::Instance();

        const auto& tuning_config = arg.get_program().get_options().get<build_option_type::tuning_config>();

        if (tuning_config->config.mode == tuning_mode::tuning_tune_and_cache ||
            tuning_config->config.mode == tuning_mode::tuning_retune_and_cache) {
            fuse_optional_params.tuningParams.runner =
                std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), arg.get_program().get_id(), true);
        }

        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(fuse_params, fuse_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto fuse = new fused_conv_bn_scale_gpu(arg, best_kernels[0]);

        return fuse;
    }
};

namespace detail {

attach_fused_conv_bn_scale_gpu::attach_fused_conv_bn_scale_gpu() {
    implementation_map<fused_conv_bn_scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                                 fused_conv_bn_scale_gpu::create);
    implementation_map<fused_conv_bn_scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                                 fused_conv_bn_scale_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
