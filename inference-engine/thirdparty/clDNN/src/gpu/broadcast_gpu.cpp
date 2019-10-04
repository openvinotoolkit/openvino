// Copyright (c) 2018-2019 Intel Corporation
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

#include "broadcast_inst.h"

#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "broadcast/broadcast_kernel_selector.h"
#include "broadcast/broadcast_kernel_base.h"
#include "error_handler.h"

namespace cldnn {
namespace gpu {

struct broadcast_gpu : typed_primitive_gpu_impl<broadcast> {
    using parent = typed_primitive_gpu_impl<broadcast>;
    using parent::parent;

    static primitive_impl* create(const broadcast_node& arg) {
        auto bc_params = get_default_params<kernel_selector::broadcast_params>(arg, 1);
        auto bc_optional_params =
            get_default_optional_params<kernel_selector::broadcast_optional_params>(arg.get_program());

        const auto format = arg.get_output_layout().format;
        size_t max_axes_num;
        if (format == format::bfzyx)
            max_axes_num = 5;
        else
            max_axes_num = 4;

        const auto& broadcast_axes = arg.get_primitive()->broadcast_axes;
        uint16_t index = (uint16_t)0;
        uint16_t input_index = (uint16_t)broadcast_axes.size();

        // bfyx, bfzyx format
        for (size_t i = 0; i < max_axes_num; ++i) {
            if (std::find(broadcast_axes.begin(), broadcast_axes.end(), i) != broadcast_axes.end()) {
                bc_params.input_order.push_back(index);
                ++index;
            } else {
                bc_params.input_order.push_back(input_index);
                ++input_index;
            }
        }

        auto& kernel_selector = kernel_selector::broadcast_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(bc_params, bc_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new broadcast_gpu(arg, best_kernels[0]);
    }
};

namespace detail {

attach_broadcast_gpu::attach_broadcast_gpu() {
    auto val_fw = broadcast_gpu::create;

    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);
    implementation_map<broadcast>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
