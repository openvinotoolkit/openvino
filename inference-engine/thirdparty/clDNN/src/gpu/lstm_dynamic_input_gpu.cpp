/*
// Copyright (c) 2019 Intel Corporation
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

#include "lstm_dynamic_input_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "lstm_dynamic/lstm_dynamic_input_kernel_selector.h"
#include "lstm_dynamic/lstm_dynamic_input_kernel_base.h"
#include "network_impl.h"
#include "error_handler.h"

namespace cldnn {
namespace gpu {

struct lstm_dynamic_input_gpu : typed_primitive_gpu_impl<lstm_dynamic_input> {
    using parent = typed_primitive_gpu_impl<lstm_dynamic_input>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<lstm_dynamic_input>& instance,
                                                        int32_t) const override {
        kernel::kernel_arguments_data args;
        args.inputs = { (memory_impl::cptr) &instance.input_memory(), (memory_impl::cptr) &instance.dyn_length_memory()};
        args.output = (memory_impl::cptr) &instance.output_memory();
        args.weights = (memory_impl::cptr) &instance.weights_memory();
        args.bias = (memory_impl::cptr) (instance.bias_term() ? &instance.bias_memory() : nullptr);
        return args;
    }

public:
    static primitive_impl* create(const lstm_dynamic_input_node& arg) {
        auto dlstm_input_params = get_default_params<kernel_selector::lstm_dynamic_input_params>(arg);

        const auto& weights_layout = arg.weights().get_output_layout();
        dlstm_input_params.weights = convert_weights_tensor(weights_layout);

        if (arg.bias_term()) {
            const auto& bias_layout = arg.bias().get_output_layout();
            dlstm_input_params.bias.push_back(convert_data_tensor(bias_layout));
        }

        // dyn length
        const auto& dyn_length_tensor = arg.dyn_length().get_output_layout();
        dlstm_input_params.inputs.push_back(convert_data_tensor(dyn_length_tensor));

        dlstm_input_params.direction = arg.direction();

        // finially get best kernel
        auto lstm_dynamic_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::lstm_dynamic_input_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::lstm_dynamic_input_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(dlstm_input_params, lstm_dynamic_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto lstm_dynamic = new lstm_dynamic_input_gpu(arg, best_kernels[0]);

        return lstm_dynamic;
    }
};

namespace detail {

attach_lstm_dynamic_input_gpu::attach_lstm_dynamic_input_gpu() {
    auto val_fw = lstm_dynamic_input_gpu::create;

    implementation_map<lstm_dynamic_input>::add({
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw},
    });
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
