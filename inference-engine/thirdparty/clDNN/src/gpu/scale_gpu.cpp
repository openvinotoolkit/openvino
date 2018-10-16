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

#include "scale_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn { namespace gpu {


struct scale_gpu : typed_primitive_gpu_impl<scale>
{
    using parent = typed_primitive_gpu_impl<scale>;
    using parent::parent;

protected:

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<scale>& instance, int32_t) const override
    {
        kernel::kernel_arguments_data args;
        args.inputs = { &instance.input_memory(), &instance.scale_memory() };
        args.output = &instance.output_memory();

        if (_outer.bias_term())
        {
            args.inputs.push_back(&instance.bias_memory());
        }
        return args;
    }

public:

    static primitive_impl* create(const scale_node& arg) 
    { 
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params = get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.inputs.push_back(convert_data_tensor(arg.scale_in().get_output_layout()));

        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Buffer(0), kernel_selector::eltwise_params::InputType::Buffer(1) },
            kernel_selector::eltwise_mode::MUL });

        if (arg.bias_term())
        {
            ew_params.inputs.push_back(convert_data_tensor(arg.bias().get_output_layout()));
            ew_params.eltwiseParams.operations.push_back({
                { kernel_selector::eltwise_params::InputType::Intermediate(0), kernel_selector::eltwise_params::InputType::Buffer(2) },
                kernel_selector::eltwise_mode::ADD });
        }

        ew_params.eltwiseParams.layoutBased = true;

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto scale = new scale_gpu(arg, best_kernels[0]);

        return scale;
    }
};

namespace {
    struct attach {
        attach() {
            auto val_fw = scale_gpu::create;

            implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
            implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
            implementation_map<scale>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
