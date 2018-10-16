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

#include "fully_connected_grad_weights_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "fully_connected_grad_weights/fully_connected_grad_weights_kernel_selector.h"
#include "fully_connected_grad_weights/fully_connected_grad_weights_kernel_base.h"
#include "api/CPP/fully_connected_grad_weights.hpp"

namespace cldnn { namespace gpu {

struct fully_connected_grad_weights_gpu : typed_primitive_gpu_impl<fully_connected_grad_weights>
{
    using parent = typed_primitive_gpu_impl<fully_connected_grad_weights>;
    using parent::parent;

protected:

    virtual bool validate(typed_primitive_inst<fully_connected_grad_weights>& instance) const override
    {
        bool res = parent::validate(instance);

        if (instance.use_momentum())
        {
            CLDNN_ERROR_LAYOUT_MISMATCH(_outer.id(), "Filter memory", instance.weights_memory().get_layout(), "previous weights grad memory", _outer.prev_weights_grad().get_output_layout(), "");
            CLDNN_ERROR_LAYOUT_MISMATCH(_outer.id(), "Bias memory", instance.bias_memory().get_layout(), "previous bias grad memory", _outer.prev_bias_grad().get_output_layout(), "");
        }

        return res;
    }

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<fully_connected_grad_weights>& instance, int32_t) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, 1);
        args.weights = &instance.weights_memory();
        args.bias = instance.bias_term() ? &instance.bias_memory() : nullptr;
        args.prev_weights_grad = instance.use_momentum() ? &instance.prev_weights_grad() : nullptr;
        args.prev_bias_grad = instance.bias_term() ? instance.use_momentum() ? &instance.prev_bias_grad() : nullptr : nullptr;

        args.lr = instance.get_network().get_learning_rate();

        return args;
    }

public:

    static primitive_impl* create(const fully_connected_grad_weights_node& arg)
    {
        auto fully_connected_grad_weights_params = get_default_learning_params<kernel_selector::fully_connected_grad_weights_params>(arg);
        auto fully_connected_grad_weights_optional_params = get_default_learning_optional_params<kernel_selector::fully_connected_grad_weights_optional_params>(arg.get_program());

        fully_connected_grad_weights_params.gradient = true;
        fully_connected_grad_weights_params.inputs.push_back(convert_data_tensor(arg.get_dependency(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::fully_connected_grad_weights_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(fully_connected_grad_weights_params, fully_connected_grad_weights_optional_params);
        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto fully_connected_grad_weights = new fully_connected_grad_weights_gpu(arg, best_kernels[0]);

        return fully_connected_grad_weights;
    };
};


namespace {
    struct attach {
        attach() {
            auto val_fw = fully_connected_grad_weights_gpu::create;

            implementation_map<fully_connected_grad_weights>::add({
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