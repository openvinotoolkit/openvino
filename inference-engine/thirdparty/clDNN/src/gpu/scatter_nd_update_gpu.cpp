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

#include "scatter_nd_update_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "scatter_update/scatter_nd_update_kernel_selector.h"
#include "scatter_update/scatter_nd_update_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct scatter_nd_update_gpu : typed_primitive_gpu_impl<scatter_nd_update> {
    using parent = typed_primitive_gpu_impl<scatter_nd_update>;
    using parent::parent;

public:
    static primitive_impl* create(const scatter_nd_update_node& arg) {
        auto scatter_nd_update_params = get_default_params<kernel_selector::scatter_nd_update_params>(arg);
        auto scatter_nd_update_optional_params =
            get_default_optional_params<kernel_selector::scatter_nd_update_optional_params>(arg.get_program());

        scatter_nd_update_params.indices_rank = arg.get_primitive()->indices_rank;

        scatter_nd_update_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        scatter_nd_update_params.inputs.push_back(convert_data_tensor(arg.input(2).get_output_layout()));

        auto& kernel_selector = kernel_selector::scatter_nd_update_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(scatter_nd_update_params, scatter_nd_update_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto scatter_nd_update = new scatter_nd_update_gpu(arg, best_kernels[0]);

        return scatter_nd_update;
    }
};

namespace detail {

attach_scatter_nd_update_gpu::attach_scatter_nd_update_gpu() {
    auto val_fw = scatter_nd_update_gpu::create;
    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);

    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);

    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<scatter_nd_update>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
