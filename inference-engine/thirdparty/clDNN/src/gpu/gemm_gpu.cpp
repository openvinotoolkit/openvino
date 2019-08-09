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

#include "gemm_inst.h"

#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "gemm/gemm_kernel_selector.h"
#include "gemm/gemm_kernel_base.h"
#include "error_handler.h"

namespace cldnn {
namespace gpu {

struct gemm_gpu : typed_primitive_gpu_impl<gemm> {
    using parent = typed_primitive_gpu_impl<gemm>;
    using parent::parent;

public:
    static primitive_impl* create(const gemm_node& arg) {
        auto gemm_params = get_default_params<kernel_selector::gemm_params>(arg, 1);
        auto gemm_optional_params =
            get_default_optional_params<kernel_selector::gemm_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            gemm_params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }

        auto desc = arg.get_primitive();
        gemm_params.alpha = desc->alpha;
        gemm_params.beta = desc->beta;
        gemm_params.transpose_input0 = desc->transpose_input0;
        gemm_params.transpose_input1 = desc->transpose_input1;

        auto& kernel_selector = kernel_selector::gemm_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gemm_params, gemm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new gemm_gpu(arg, best_kernels[0]);
    }
};

namespace {
struct attach {
    attach() {
        auto val_fw = gemm_gpu::create;
        implementation_map<gemm>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
        implementation_map<gemm>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        implementation_map<gemm>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
        implementation_map<gemm>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
        implementation_map<gemm>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
        implementation_map<gemm>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    }
    ~attach() = default;
};

attach attach_impl;

}  // namespace
}  // namespace gpu
}  // namespace cldnn
