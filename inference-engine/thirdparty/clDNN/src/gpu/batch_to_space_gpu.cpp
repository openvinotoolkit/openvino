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

#include "batch_to_space_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "batch_to_space/batch_to_space_kernel_selector.h"
#include "batch_to_space/batch_to_space_kernel_ref.h"
#include "error_handler.h"
#include "data_inst.h"
#include <vector>

using namespace cldnn;

namespace cldnn {
namespace gpu {
struct batch_to_space_gpu : typed_primitive_gpu_impl<batch_to_space> {
    using parent = typed_primitive_gpu_impl<batch_to_space>;
    using parent::parent;

public:
    static primitive_impl* create(const batch_to_space_node& arg) {
        auto batch_to_space_params = get_default_params<kernel_selector::batch_to_space_params>(arg);
        auto batch_to_space_optional_params =
            get_default_optional_params<kernel_selector::batch_to_space_optional_params>(arg.get_program());

        // Getting data from constant inputs. There are 3 args: block_shape, crops_begin, crops_end
        for (size_t i = 1; i < arg.get_dependencies().size(); ++i) {
            auto& input = arg.get_dependency(i).as<data>();
            auto& mem = input.get_attached_memory();
            int32_t* data = static_cast<int32_t*>(mem.lock());
            std::vector<int32_t> sizes = std::vector<int32_t>(data, data + input.get_output_layout().count());

            batch_to_space_params.bts_params.push_back(sizes);
            mem.unlock();
        }

        auto& kernel_selector = kernel_selector::batch_to_space_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(batch_to_space_params, batch_to_space_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto batch_to_space = new batch_to_space_gpu(arg, best_kernels[0]);

        return batch_to_space;
    }
};

namespace detail {

attach_batch_to_space_gpu::attach_batch_to_space_gpu() {
    auto val_fw = batch_to_space_gpu::create;
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
