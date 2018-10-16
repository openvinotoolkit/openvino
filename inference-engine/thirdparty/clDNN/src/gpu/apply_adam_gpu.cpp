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

#include "apply_adam_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"

namespace cldnn { namespace gpu {

struct apply_adam_gpu : typed_primitive_gpu_impl<apply_adam>
{
    using parent = typed_primitive_gpu_impl<apply_adam>;
    using parent::parent;

protected:

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<apply_adam>& instance, int32_t) const override
    {
        kernel::kernel_arguments_data args;

        args.inputs = { &instance.input_memory(), &instance.m_memory(), &instance.v_memory(), &instance.beta1_power_memory(), &instance.beta2_power_memory() };
        args.output = &instance.output_memory();

        return args;
    }

public:

    static primitive_impl* create(const apply_adam_node &arg) 
    { 
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params = get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());
        const float lr = arg.get_primitive()->lr;
        const float beta1 = arg.get_primitive()->beta1;
        const float beta2 = arg.get_primitive()->beta2;
        const float epsilon =
            (arg.input().get_output_layout().data_type == data_types::f16) ?
            std::max(0.00007f, arg.get_primitive()->epsilon) : // prevent underflow if the epsilon is too small for fp16
            arg.get_primitive()->epsilon;

        ew_params.inputs.push_back(convert_data_tensor(arg.m().get_output_layout()));
        ew_params.inputs.push_back(convert_data_tensor(arg.v().get_output_layout()));
        ew_params.inputs.push_back(convert_data_tensor(arg.beta1_power().get_output_layout()));
        ew_params.inputs.push_back(convert_data_tensor(arg.beta2_power().get_output_layout()));

        //lr_t = lr * sqrt(1 - pow(beta2, t_f)) / (1 - pow(beta1, t_f))
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Scalar(1), kernel_selector::eltwise_params::InputType::Buffer(3) },
            kernel_selector::eltwise_mode::SUB });

        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Scalar(1), kernel_selector::eltwise_params::InputType::Buffer(4) },
            kernel_selector::eltwise_mode::SUB });

        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(1) },
            kernel_selector::eltwise_mode::SQRT });

        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(2), kernel_selector::eltwise_params::InputType::Scalar(lr) },
            kernel_selector::eltwise_mode::MUL });

        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(3), kernel_selector::eltwise_params::InputType::Intermediate(0) },
            kernel_selector::eltwise_mode::DIV });

        //m_t = beta1 * m_f + (1 - beta1) * input_grad
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Scalar(beta1), kernel_selector::eltwise_params::InputType::Buffer(1) },
            kernel_selector::eltwise_mode::MUL });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Scalar(1), kernel_selector::eltwise_params::InputType::Scalar(beta1) },
            kernel_selector::eltwise_mode::SUB });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(6), kernel_selector::eltwise_params::InputType::Buffer(0) },
            kernel_selector::eltwise_mode::MUL });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(5), kernel_selector::eltwise_params::InputType::Intermediate(7) },
            kernel_selector::eltwise_mode::ADD });

        //save the result in m mutable_data primitive
        ew_params.eltwiseParams.updateInputIds.push_back({ 1, 8 });
        
        ////v_t = beta2 * v_f + (1 - beta2) * input_grad * input_grad
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Scalar(beta2), kernel_selector::eltwise_params::InputType::Buffer(2) },
            kernel_selector::eltwise_mode::MUL });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Scalar(1), kernel_selector::eltwise_params::InputType::Scalar(beta2) },
            kernel_selector::eltwise_mode::SUB });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(10), kernel_selector::eltwise_params::InputType::Buffer(0) },
            kernel_selector::eltwise_mode::MUL });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(11), kernel_selector::eltwise_params::InputType::Buffer(0) },
            kernel_selector::eltwise_mode::MUL });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(9), kernel_selector::eltwise_params::InputType::Intermediate(12) },
            kernel_selector::eltwise_mode::ADD });
        
        //save the result in v mutable_data primitive
        ew_params.eltwiseParams.updateInputIds.push_back({ 2, 13 });

        ////result = var - lr_t * m_t / (sqrt(v_t) + epsilon)
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(13) },
            kernel_selector::eltwise_mode::SQRT });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(14), kernel_selector::eltwise_params::InputType::Scalar(epsilon) },
            kernel_selector::eltwise_mode::ADD });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(4), kernel_selector::eltwise_params::InputType::Intermediate(8) },
            kernel_selector::eltwise_mode::MUL });
        
        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Intermediate(16), kernel_selector::eltwise_params::InputType::Intermediate(15) },
            kernel_selector::eltwise_mode::DIV });

        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::OutBuffer(), kernel_selector::eltwise_params::InputType::Intermediate(17) },
            kernel_selector::eltwise_mode::SUB });

        ew_params.eltwiseParams.layoutBased = true;

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto norm = new apply_adam_gpu(arg, best_kernels[0]);

        return norm;
    };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = apply_adam_gpu::create;

            implementation_map<apply_adam>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<apply_adam>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<apply_adam>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<apply_adam>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
            implementation_map<apply_adam>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
            implementation_map<apply_adam>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
