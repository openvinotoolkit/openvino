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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "lstm_gemm_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "lstm/lstm_gemm_kernel_selector.h"
#include "lstm/lstm_gemm_kernel_base.h"
#include "network_impl.h"
#include "error_handler.h"
#include "lstm/lstm_gemm_kernel_selector.h"
#include "lstm/lstm_gemm_kernel_base.h"


namespace cldnn { namespace gpu {

struct lstm_gemm_gpu : typed_primitive_gpu_impl<lstm_gemm>
{
    using parent = typed_primitive_gpu_impl<lstm_gemm>;
    using parent::parent;

protected:

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<lstm_gemm>& instance, int32_t) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, 0);

        args.output     = &instance.output_memory();
        args.weights    = &instance.weights_memory();
        args.recurrent  = &instance.recurrent_memory();
        args.bias       = instance.bias_term() ? &instance.bias_memory() : nullptr;
        args.hidden     = instance.hidden_term() ? &instance.hidden_memory() : nullptr;

        return args;
    }

    virtual bool validate(typed_primitive_inst<lstm_gemm>& instance) const override
    {
        bool res = parent::validate(instance);

        return res;
    }
public:

    static primitive_impl* create(const lstm_gemm_node& arg)
    {
        const auto& weights_layout   = arg.weights().get_output_layout();

        auto lstm_gemm_params = get_default_params<kernel_selector::lstm_gemm_params>(arg);
        lstm_gemm_params.weights = convert_data_tensor(weights_layout);


        if (arg.bias_term())
        {
            const auto& bias_layout = arg.bias().get_output_layout();
            lstm_gemm_params.SetBias(convert_data_tensor(bias_layout));
        }
        if (arg.hidden_term())
        {
            const auto& recurrent_layout = arg.recurrent().get_output_layout();
            lstm_gemm_params.recurrent = convert_data_tensor(recurrent_layout);

            const auto& hidden_layout = arg.hidden().get_output_layout();
            lstm_gemm_params.SetHidden(convert_data_tensor(hidden_layout));
        }
        lstm_gemm_params.direction = arg.direction();

        auto lstm_gemm_optional_params = get_default_optional_params<kernel_selector::lstm_gemm_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::lstm_gemm_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(lstm_gemm_params, lstm_gemm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto lstm_gemm = new lstm_gemm_gpu(arg, best_kernels[0]);

        return lstm_gemm;
    };
};


namespace {
    struct attach {
        attach() {
            auto val_fw = lstm_gemm_gpu::create;

            implementation_map<lstm_gemm>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
