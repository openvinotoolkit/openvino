/*
// Copyright (c) 2021 Intel Corporation
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

#include "gather_elements_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "gather/gather_elements_kernel_selector.h"
#include "gather/gather_elements_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {
kernel_selector::gather_elements_axis convert_axis(gather_elements::gather_elements_axis axis) {
    switch (axis) {
        case gather_elements::along_x:
            return kernel_selector::gather_elements_axis::X;
        case gather_elements::along_y:
            return kernel_selector::gather_elements_axis::Y;
        case gather_elements::along_z:
            return kernel_selector::gather_elements_axis::Z;
        case gather_elements::along_w:
            return kernel_selector::gather_elements_axis::W;
        case gather_elements::along_f:
            return kernel_selector::gather_elements_axis::FEATURE;
        case gather_elements::along_b:
            return kernel_selector::gather_elements_axis::BATCH;
        default:
            return kernel_selector::gather_elements_axis::X;
    }
}

struct gather_elements_gpu : typed_primitive_gpu_impl<gather_elements> {
    using parent = typed_primitive_gpu_impl<gather_elements>;
    using parent::parent;

public:
    static primitive_impl* create(const gather_elements_node& arg) {
        auto gather_elements_params = get_default_params<kernel_selector::gather_elements_params>(arg);
        auto gather_elements_optional_params =
            get_default_optional_params<kernel_selector::gather_elements_optional_params>(arg.get_program());

        gather_elements_params.axis = convert_axis(arg.get_primitive()->axis);

        gather_elements_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::gather_elements_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gather_elements_params, gather_elements_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto gather_elements = new gather_elements_gpu(arg, best_kernels[0]);

        return gather_elements;
    }
};

namespace detail {

attach_gather_elements_gpu::attach_gather_elements_gpu() {
    auto val_fw = gather_elements_gpu::create;
    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);

    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);

    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<gather_elements>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
