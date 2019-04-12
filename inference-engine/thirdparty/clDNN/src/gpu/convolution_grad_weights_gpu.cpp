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

#include "convolution_grad_weights_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "network_impl.h"
#include "kernel_selector_helper.h"
#include "convolution_grad_weights/convolution_grad_weights_kernel_selector.h"
#include "convolution_grad_weights/convolution_grad_weights_kernel_base.h"
namespace cldnn { namespace gpu {

struct convolution_grad_weights_gpu : typed_primitive_gpu_impl<convolution_grad_weights>
{
    using parent = typed_primitive_gpu_impl<convolution_grad_weights>;
    using parent::parent;

protected:

    virtual bool validate_impl(const typed_primitive_inst<convolution_grad_weights>& instance) const override
    {
        bool res = true;

        CLDNN_ERROR_NOT_EQUAL(_outer.id(), "convolution_grad_weights filling value", _outer.get_output_layout().data_padding.filling_value(), "padding mode", 0.0f, "Unknown padding mode in convolution_grad_weights.");
        // Check whether all memory elements use the same unit type (FP16 or FP32).
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input grad memory", instance.input_memory().get_layout().data_type, "output memory", instance.output_memory().get_layout().data_type, "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.input_memory(1).get_layout().data_type, "output memory", instance.output_memory().get_layout().data_type, "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Fp32", data_types::f32, "filter memory", instance.weights_memory(0).get_layout().data_type, "");

        if (instance.use_momentum())
        {
            CLDNN_ERROR_LAYOUT_MISMATCH(_outer.id(), "Filter memory", instance.weights_memory(0).get_layout(), "previous weights grad memory", _outer.prev_weights_grad(0).get_output_layout(), "");
            CLDNN_ERROR_LAYOUT_MISMATCH(_outer.id(), "Bias memory", instance.bias_memory(0).get_layout(), "previous bias grad memory", _outer.prev_bias_grad(0).get_output_layout(), "");
        }

        return res;
    }

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<convolution_grad_weights>& instance, int32_t split) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights    = &instance.weights_memory(split);
        args.bias       = instance.bias_term() ? &instance.bias_memory(split) : nullptr;
        args.prev_weights_grad = instance.use_momentum() ? &instance.prev_weights_grad(split) : nullptr;
        args.prev_bias_grad = instance.bias_term() ? instance.use_momentum() ? &instance.prev_bias_grad(split) : nullptr : nullptr;
        args.lr         = instance.get_network().get_learning_rate();

        return args;
    }

    virtual int32_t get_split() const override
    { 
        return _outer.get_split(); 
    }

public:

    static primitive_impl* create(const convolution_grad_weights_node& arg)
    {
        const auto& primitive = arg.get_primitive();
        const auto& weights_layout = arg.weights(0).get_output_layout();

        switch (weights_layout.fused_format())
        {
        case fuse(data_types::f32, format::bfyx):
        case fuse(data_types::f32, format::yxfb):
        case fuse(data_types::f16, format::bfyx):
        case fuse(data_types::f16, format::yxfb):
            break;
        default:
            throw std::runtime_error("convolution_grad_weights weights format unsupported");
        }

        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& stride = primitive->stride;
#if 0 // TODO: support dilation
        const auto& dilation = primitive->dilation;
#else
        const tensor dilation = {0,0,1,1};
#endif
        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        const auto output_grad_w = arg.output_grad_w();

        const auto& input_offset = primitive->input_offset;

        auto conv_grad_weights_params = get_default_learning_params<kernel_selector::convolution_grad_weights_params>(arg, depthwise_separable_opt ? 1 : split);
        auto conv_grad_weights_optional_params = get_default_learning_optional_params<kernel_selector::convolution_grad_weights_optional_params>(arg.get_program());

        conv_grad_weights_params.depthwise_separable_opt = depthwise_separable_opt;
        conv_grad_weights_params.output_grad_w = output_grad_w;

        conv_grad_weights_params.gradient = true;
        conv_grad_weights_params.inputs.push_back(convert_data_tensor(arg.get_dependency(1).get_output_layout()));

        conv_grad_weights_params.split = split;
        conv_grad_weights_params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
        };

        conv_grad_weights_params.padding = {
            (uint32_t)std::max(-input_offset.spatial[0], 0),
            (uint32_t)std::max(-input_offset.spatial[1], 0)
        };

        conv_grad_weights_params.stride = {
            (uint32_t)stride.spatial[0],
            (uint32_t)stride.spatial[1]
        };

        conv_grad_weights_params.dilation = {
            (uint32_t)dilation.spatial[0],
            (uint32_t)dilation.spatial[1]
        };

        auto& kernel_selector = kernel_selector::convolution_grad_weights_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(conv_grad_weights_params, conv_grad_weights_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto deconv = new convolution_grad_weights_gpu(arg, best_kernels[0]);

        return deconv;
    }
};

namespace{
    struct attach {
        attach() {
            implementation_map<convolution_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), convolution_grad_weights_gpu::create);
            implementation_map<convolution_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), convolution_grad_weights_gpu::create);
            implementation_map<convolution_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), convolution_grad_weights_gpu::create);
            implementation_map<convolution_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), convolution_grad_weights_gpu::create);
            implementation_map<convolution_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), convolution_grad_weights_gpu::create);
            implementation_map<convolution_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), convolution_grad_weights_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
