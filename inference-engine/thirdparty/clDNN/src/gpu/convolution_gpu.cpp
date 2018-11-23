/*
// Copyright (c) 2016 Intel Corporation
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

#include "convolution_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "convolution/convolution_kernel_selector.h"
#include "convolution/convolution_params.h"

namespace cldnn { namespace gpu {

struct convolution_gpu : typed_primitive_gpu_impl<convolution>
{
    using parent = typed_primitive_gpu_impl<convolution>;
    using parent::parent;

protected:

    virtual bool validate(typed_primitive_inst<convolution>& instance) const override
    {
        bool res = parent::validate(instance);

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.node.input().get_output_layout().data_type, "output memory", instance.node.get_output_layout().data_type, "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.node.input().get_output_layout().data_type, "filter memory", instance.weights_memory(0).get_layout().data_type, "");

        return res;
    }

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<convolution>& instance, int32_t split) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights              = &instance.weights_memory(split);
        args.bias                 = instance.bias_term() ? &instance.bias_memory(split) : nullptr;
        args.weights_quantization_factors = instance.weights_quantization_factors_term() ? &instance.weights_quantization_factors_memory(split) : nullptr;
        args.output_calibration_factors = instance.output_calibration_factors_term() ? &instance.output_calibration_factors_memory(split) : nullptr;
        return args;
    }

    virtual int32_t get_split() const override
    { 
        return _outer.get_split(); 
    }

public:

    static primitive_impl* create(const convolution_node &arg)
    {
        const auto& primitive       = arg.get_primitive();
        const auto& input_layout    = arg.input().get_output_layout();
        const auto& weights_layout  = arg.weights(0).get_output_layout();
        const auto& weights_size    = weights_layout.size;

        const auto& split           = primitive->split();
        const auto& stride          = primitive->stride;
        const auto& dilation        = primitive->dilation;
        const auto& input_offset    = primitive->input_offset;

        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        const auto actual_split = depthwise_separable_opt ? (decltype(split))1 : split;

        const auto transposed = arg.get_transposed();

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        auto conv_params = get_weights_bias_default_params<kernel_selector::convolution_params>(arg, actual_split);
        auto conv_optional_params = get_default_weights_bias_optional_params<kernel_selector::convolution_optional_params>(arg.get_program());

        const auto additional_offset = tensor::max(input_offset, 0);
        if (additional_offset != 0)
        {
            conv_params.inputs[0] = convert_data_tensor(input_layout, actual_split, additional_offset);
        }

        if(primitive->with_activation)
            convert_activation_func_params(primitive, conv_params);

        conv_params.depthwiseSeparableOpt = depthwise_separable_opt;
        conv_params.transposed = transposed;

        conv_params.split = split;
        conv_params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
        };

        conv_params.padding = {
            (uint32_t)std::max(-input_offset.spatial[0], 0),
            (uint32_t)std::max(-input_offset.spatial[1], 0)
        };

        conv_params.stride = {
            (uint32_t)stride.spatial[0],
            (uint32_t)stride.spatial[1]
        };
        conv_params.dilation = {
            (uint32_t)dilation.spatial[0],
            (uint32_t)dilation.spatial[1]
        };
        
        if (primitive->weights_quantization_factors.size() > 0)
        {
            conv_params.int8_quantization = true;
            conv_params.weights_quantization_factors.push_back(convert_data_tensor(arg.weights_quantization_factors().get_output_layout()).FlattenFeatureAndSpatials());
            conv_params.input_quantization_factor = arg.get_input_qf();

            if (primitive->output_calibration_factors.size() > 0)
            {
                conv_params.output_calibration = true;
                conv_params.output_calibration_factors.push_back(convert_data_tensor(arg.output_calibration_factors().get_output_layout()).FlattenFeatureAndSpatials());
            }
            else
                conv_params.output_quantization_factor = arg.get_output_qf();
        }

        auto& kernel_selector = kernel_selector::convolution_kernel_selector::Instance();

        const auto& tuning_config = arg.get_program().get_options().get<build_option_type::tuning_config>();

        if (tuning_config->config.mode == tuning_mode::tuning_tune_and_cache)
        {
            conv_optional_params.tuningParams.runner = std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), true);
        }

        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);
		
        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto conv = new convolution_gpu(arg, best_kernels[0]);

        return conv;
    }
};

namespace{
    struct attach {
        attach() {
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::winograd_2x3_s1_data), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::winograd_2x3_s1_data), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bf8_xy16), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bf8_xy16), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), convolution_gpu::create);
            // MMAD
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf_af32), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::fs_bs_yx_bsv4_fsv32), convolution_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
