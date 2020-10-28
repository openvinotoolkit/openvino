// Copyright (c) 2019-2020 Intel Corporation
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

#include "gather_tree_inst.h"

#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "gather_tree/gather_tree_kernel_selector.h"
#include "gather_tree/gather_tree_kernel_base.h"
#include "error_handler.h"
namespace cldnn {
namespace gpu {

struct gather_tree_gpu : typed_primitive_gpu_impl<gather_tree> {
    using parent = typed_primitive_gpu_impl<gather_tree>;
    using parent::parent;

    static primitive_impl* create(const gather_tree_node& arg) {
        auto b_params = get_default_params<kernel_selector::gather_tree_params>(arg, 1);
        auto b_optional_params = get_default_optional_params<kernel_selector::gather_tree_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.get_dependencies().size(); i++) {
            b_params.inputs.push_back(convert_data_tensor(arg.get_dependency(i).get_output_layout(), 1));
        }
        auto desc = arg.get_primitive();

        auto& kernel_selector = kernel_selector::gather_tree_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(b_params, b_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
            "Best_kernel.empty()",
            best_kernels.empty(),
            "Cannot find a proper kernel with this arguments");

        return new gather_tree_gpu(arg, best_kernels[0]);
    }
};
namespace detail {
    attach_gather_tree_gpu::attach_gather_tree_gpu() {
            auto val_fw = gather_tree_gpu::create;

            implementation_map<gather_tree>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::yxfb), val_fw);
            implementation_map<gather_tree>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
            implementation_map<gather_tree>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::byxf), val_fw);

            implementation_map<gather_tree>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<gather_tree>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<gather_tree>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
        }

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
