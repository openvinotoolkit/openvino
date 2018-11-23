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

#include "eltwise_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"

namespace cldnn { namespace gpu {

namespace
{
    inline kernel_selector::eltwise_mode convect_to_eltwise_mode(eltwise_mode mode)
    {
        switch (mode)
        {
        case eltwise_mode::sum:  return kernel_selector::eltwise_mode::ADD;
        case eltwise_mode::sub:  return kernel_selector::eltwise_mode::SUB;
        case eltwise_mode::max:  return kernel_selector::eltwise_mode::MAX;
        case eltwise_mode::prod: return kernel_selector::eltwise_mode::MUL;
        case eltwise_mode::div: return kernel_selector::eltwise_mode::DIV;
        case eltwise_mode::min: return kernel_selector::eltwise_mode::MIN;
        case eltwise_mode::pow: return kernel_selector::eltwise_mode::POW;
        case eltwise_mode::mod: return kernel_selector::eltwise_mode::MODULU;
        default:
            return kernel_selector::eltwise_mode::ADD;
        }
    }
}

struct eltwise_gpu : typed_primitive_gpu_impl<eltwise>
{
    using parent = typed_primitive_gpu_impl<eltwise>;
    using parent::parent;
protected:
    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<eltwise>& instance, int32_t split) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.output_calibration_factors = instance.output_calibration_factors_term() ? &instance.output_calibration_factors_memory() : nullptr;
        return args;
    }

public:
    static primitive_impl* create(const eltwise_node& arg) 
    { 
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params = get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.inputs_count(); i++)
        {
            ew_params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }

        const auto& primitive = arg.get_primitive();
        if(primitive->with_activation)
            convert_activation_func_params(primitive, ew_params);

        ew_params.operations.push_back({ 
            { kernel_selector::eltwise_params::InputType::Buffer(0), kernel_selector::eltwise_params::InputType::Buffer(1) },
            convect_to_eltwise_mode(primitive->mode) });

        for (uint32_t i = 2; i < static_cast<uint32_t>(arg.inputs_count()); i++)
        {
            ew_params.operations.push_back({{ kernel_selector::eltwise_params::InputType::Intermediate(i-2),
                                                            kernel_selector::eltwise_params::InputType::Buffer(i) },
                                                            convect_to_eltwise_mode(primitive->mode) });
        }

        if (primitive->mode == eltwise_mode::sum)
        {
            ew_params.coefficients = primitive->coefficients;
        }

        for (size_t i = 0; i < ew_params.inputs.size(); i++)
        {
            if (!ew_params.inputs[i].SameDims(ew_params.output))
                ew_params.layoutBased = true;
        }

        if (primitive->output_calibration_factors.size() > 0 || primitive->output_quantization_factor != 1.0f)
        {
            ew_params.int8_quantization = true;

            if (primitive->output_calibration_factors.size() > 0)
            {
                ew_params.output_calibration = true;
                ew_params.output_calibration_factors.push_back(convert_data_tensor(arg.output_calibration_factors().get_output_layout()).FlattenFeatureAndSpatials());
            }
            else
                ew_params.output_quantization_factor = arg.get_output_qf();
        }

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto eltwise = new eltwise_gpu(arg, best_kernels[0]);

        return eltwise;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<eltwise>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i32, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i64, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i32, format::byxf), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i64, format::byxf), eltwise_gpu::create },
                // MMAD
                { std::make_tuple(engine_types::ocl, data_types::i8, format::byxf_af32), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::i8, format::fs_bs_yx_bsv4_fsv32), eltwise_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
