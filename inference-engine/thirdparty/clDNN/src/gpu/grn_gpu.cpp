/*
// Copyright (c) 2020 Intel Corporation
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

#include "grn_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "grn/grn_kernel_selector.h"
#include "grn/grn_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct grn_gpu : typed_primitive_gpu_impl<grn> {
    using parent = typed_primitive_gpu_impl<grn>;
    using parent::parent;

public:
    static primitive_impl* create(const grn_node& arg) {
        auto grn_params = get_default_params<kernel_selector::grn_params>(arg);
        auto grn_optional_params = get_default_optional_params<kernel_selector::grn_optional_params>(arg.get_program());

        grn_params.bias = arg.get_primitive()->bias;

        auto& kernel_selector = kernel_selector::grn_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(grn_params, grn_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto grn = new grn_gpu(arg, best_kernels[0]);

        return grn;
    }
};

namespace detail {

attach_grn_gpu::attach_grn_gpu() {
    implementation_map<grn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), grn_gpu::create);
    implementation_map<grn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), grn_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
