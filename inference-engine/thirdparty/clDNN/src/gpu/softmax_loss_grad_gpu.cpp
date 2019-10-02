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

#include "softmax_loss_grad_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "softmax_loss_grad/softmax_loss_grad_kernel_selector.h"
#include "softmax_loss_grad/softmax_loss_grad_kernel_base.h"
#include "error_handler.h"

namespace cldnn {
namespace gpu {

struct softmax_loss_grad_gpu : typed_primitive_gpu_impl<softmax_loss_grad> {
    using parent = typed_primitive_gpu_impl<softmax_loss_grad>;
    using parent::parent;

    static primitive_impl* create(const softmax_loss_grad_node& arg) {
        auto sm_params = get_default_params<kernel_selector::softmax_loss_grad_params>(arg);
        auto sm_optional_params =
            get_default_optional_params<kernel_selector::softmax_loss_grad_optional_params>(arg.get_program());

        sm_params.gradient = true;
        sm_params.inputs.push_back(convert_data_tensor(arg.get_dependency(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::softmax_loss_grad_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(sm_params, sm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto softmax_loss_grad_node = new softmax_loss_grad_gpu(arg, best_kernels[0]);

        return softmax_loss_grad_node;
    }
};

namespace detail {

attach_softmax_loss_grad_gpu::attach_softmax_loss_grad_gpu() {
    auto val_fw = softmax_loss_grad_gpu::create;
    implementation_map<softmax_loss_grad>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                               val_fw);
    implementation_map<softmax_loss_grad>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                               val_fw);
    implementation_map<softmax_loss_grad>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                               val_fw);
    implementation_map<softmax_loss_grad>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                               val_fw);
    implementation_map<softmax_loss_grad>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                               val_fw);
    implementation_map<softmax_loss_grad>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                               val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
