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

#include "scale_grad_weights_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "scale_grad_weights/scale_grad_weights_kernel_selector.h"
#include "scale_grad_weights/scale_grad_weights_kernel_base.h"
#include "error_handler.h"
#include "network_impl.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct scale_grad_weights_gpu : typed_primitive_gpu_impl<scale_grad_weights> {
    using parent = typed_primitive_gpu_impl<scale_grad_weights>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<scale_grad_weights>& instance,
                                                        int32_t) const override {
        kernel::kernel_arguments_data args;
        args.inputs = {(memory_impl::cptr) &instance.input_memory(0), (memory_impl::cptr) &instance.input_memory(1)};
        args.output = (memory_impl::cptr) &instance.output_memory();

        args.bias = (memory_impl::cptr) (_outer.bias_term() ? &instance.bias_memory() : nullptr);
        args.weights = (memory_impl::cptr) &instance.weights_memory();

        args.prev_weights_grad = (memory_impl::cptr) (instance.use_momentum() ? &instance.prev_scale_grad() : nullptr);
        args.prev_bias_grad =
            (memory_impl::cptr) (instance.bias_term() ? instance.use_momentum() ? &instance.prev_bias_grad() : nullptr : nullptr);
        args.lr = instance.get_network().get_learning_rate();

        return args;
    }

public:
    static primitive_impl* create(const scale_grad_weights_node& arg) {
        auto scale_params = get_default_learning_params<kernel_selector::scale_grad_weights_params>(arg);
        auto scale_optional_params =
            get_default_learning_optional_params<kernel_selector::scale_grad_weights_optional_params>(
                arg.get_program());

        auto& kernel_selector = kernel_selector::scale_grad_weights_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(scale_params, scale_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto scale_grad_weights = new scale_grad_weights_gpu(arg, best_kernels[0]);

        return scale_grad_weights;
    }
};

namespace detail {

attach_scale_grad_weights_gpu::attach_scale_grad_weights_gpu() {
    auto val_fw = scale_grad_weights_gpu::create;

    implementation_map<scale_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                                val_fw);
    implementation_map<scale_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                                val_fw);
    implementation_map<scale_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                                val_fw);
    implementation_map<scale_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                                val_fw);
    implementation_map<scale_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                                val_fw);
    implementation_map<scale_grad_weights>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                                val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
