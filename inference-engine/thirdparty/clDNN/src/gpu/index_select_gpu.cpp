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

#include "index_select_inst.h"

#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "index_select/index_select_kernel_selector.h"
#include "index_select/index_select_kernel_base.h"
#include "error_handler.h"
#include <vector>

namespace cldnn {
namespace gpu {

namespace {
inline std::vector<kernel_selector::IndexSelectAxis> convert_to_index_select_axis(
    std::vector<index_select_axis_name> axes) {
    std::vector<kernel_selector::IndexSelectAxis> axes_names = {};
    for (size_t i = 0; i < axes.size(); i++) {
        switch (axes[i]) {
            case index_select_axis_name::along_b:
                axes_names.push_back(kernel_selector::IndexSelectAxis::BATCH);
                break;
            case index_select_axis_name::along_f:
                axes_names.push_back(kernel_selector::IndexSelectAxis::FEATURE);
                break;
            case index_select_axis_name::along_x:
                axes_names.push_back(kernel_selector::IndexSelectAxis::X);
                break;
            case index_select_axis_name::along_y:
                axes_names.push_back(kernel_selector::IndexSelectAxis::Y);
                break;
            default:
                axes_names.push_back(kernel_selector::IndexSelectAxis::BATCH);
                break;
        }
    }
    return axes_names;
}
}  // namespace

struct index_select_gpu : typed_primitive_gpu_impl<index_select> {
    using parent = typed_primitive_gpu_impl<index_select>;
    using parent::parent;

    static primitive_impl* create(const index_select_node& arg) {
        auto index_select_params = get_default_params<kernel_selector::index_select_params>(arg, 1);
        auto index_select_optional_params =
            get_default_optional_params<kernel_selector::index_select_optional_params>(arg.get_program());

        if (!arg.get_reverse())
            index_select_params.inputs.push_back(convert_data_tensor(arg.indices().get_output_layout()));

        index_select_params.axes = convert_to_index_select_axis(arg.get_axes());
        index_select_params.reverse = arg.get_reverse();

        auto& kernel_selector = kernel_selector::index_select_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(index_select_params, index_select_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new index_select_gpu(arg, best_kernels[0]);
    }
};

namespace detail {

attach_index_select_gpu::attach_index_select_gpu() {
    auto val_fw = index_select_gpu::create;
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                          val_fw);
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                          val_fw);
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx),
                                          val_fw);

    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                          val_fw);
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                          val_fw);
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb), val_fw);
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::yxfb), val_fw);
    implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::yxfb),
                                          val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
