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

#include "batch_norm_grad_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "batch_norm_grad/batch_norm_grad_kernel_base.h"
#include "batch_norm_grad/batch_norm_grad_kernel_selector.h"

namespace cldnn {
namespace gpu {

struct batch_norm_grad_gpu : typed_primitive_gpu_impl<batch_norm_grad> {
    using parent = typed_primitive_gpu_impl<batch_norm_grad>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<batch_norm_grad>& instance,
                                                        int32_t) const override {
        kernel::kernel_arguments_data args;

        args.inputs = {
            (memory_impl::cptr) &instance.input_memory(0),
            (memory_impl::cptr) &instance.input_memory(1),
            (memory_impl::cptr) &instance.inv_variance_memory()};
        args.output = (memory_impl::cptr) &instance.output_memory();

        return args;
    }

public:
    static primitive_impl* create(const batch_norm_grad_node& arg) {
        auto norm_params = get_default_params<kernel_selector::batch_norm_grad_params>(arg);
        auto norm_optional_params =
            get_default_optional_params<kernel_selector::batch_norm_grad_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::batch_norm_grad_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(norm_params, norm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto norm = new batch_norm_grad_gpu(arg, best_kernels[0]);

        return norm;
    }
};

namespace detail {

attach_batch_norm_grad_gpu::attach_batch_norm_grad_gpu() {
    auto val_fw = batch_norm_grad_gpu::create;

    implementation_map<batch_norm_grad>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                                val_fw);
    implementation_map<batch_norm_grad>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                                val_fw);
    implementation_map<batch_norm_grad>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                                val_fw);
    implementation_map<batch_norm_grad>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                                val_fw);
    implementation_map<batch_norm_grad>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                                val_fw);
    implementation_map<batch_norm_grad>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                                val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
