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

#include "fully_connected_grad_input_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "fully_connected_grad_input/fully_connected_grad_input_kernel_selector.h"
#include "fully_connected_grad_input/fully_connected_grad_input_kernel_base.h"
#include "api/CPP/fully_connected_grad_input.hpp"

namespace cldnn { namespace gpu {

struct fully_connected_grad_input_gpu : typed_primitive_gpu_impl<fully_connected_grad_input>
{
    using parent = typed_primitive_gpu_impl<fully_connected_grad_input>;
    using parent::parent;

protected:

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<fully_connected_grad_input>& instance, int32_t) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, 1);
        args.weights = &instance.weights_memory();

        return args;
    }

public:

    static primitive_impl* create(const fully_connected_grad_input_node& arg)
    {
        auto fully_connected_grad_input_params = get_default_params<kernel_selector::fully_connected_grad_input_params>(arg);
        auto fully_connected_grad_input_optional_params = get_default_optional_params<kernel_selector::fully_connected_grad_input_optional_params>(arg.get_program());

        const auto& weights_layout = arg.weights().get_output_layout();
        fully_connected_grad_input_params.weights = convert_weights_tensor(weights_layout);
        fully_connected_grad_input_params.gradient = true;
        fully_connected_grad_input_params.inputs.push_back(convert_data_tensor(arg.get_dependency(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::fully_connected_grad_input_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(fully_connected_grad_input_params, fully_connected_grad_input_optional_params);
        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto fully_connected_grad_input = new fully_connected_grad_input_gpu(arg, best_kernels[0]);

        return fully_connected_grad_input;
    };
};


namespace {
    struct attach {
        attach() {
            auto val_fw = fully_connected_grad_input_gpu::create;

            implementation_map<fully_connected_grad_input>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }