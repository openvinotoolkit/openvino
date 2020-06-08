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

#include "space_to_batch_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "space_to_batch/space_to_batch_kernel_selector.h"
#include "space_to_batch/space_to_batch_kernel_ref.h"
#include "error_handler.h"
#include "data_inst.h"
#include <vector>

using namespace cldnn;

namespace cldnn {
namespace gpu {
struct space_to_batch_gpu : typed_primitive_gpu_impl<space_to_batch> {
    using parent = typed_primitive_gpu_impl<space_to_batch>;
    using parent::parent;

public:
    static primitive_impl* create(const space_to_batch_node& arg) {
        auto space_to_batch_params = get_default_params<kernel_selector::space_to_batch_params>(arg);
        auto space_to_batch_optional_params =
            get_default_optional_params<kernel_selector::space_to_batch_optional_params>(arg.get_program());

        // Getting data from constant inputs. There are 3 args: block_shape, pads_begin, pads_end
        for (size_t i = 1; i < arg.get_dependencies().size(); ++i) {
            auto& input = arg.get_dependency(i).as<data>();
            auto& mem = input.get_attached_memory();
            int32_t* data = static_cast<int32_t*>(mem.lock());
            std::vector<int32_t> sizes = std::vector<int32_t>(data, data + input.get_output_layout().count());

            space_to_batch_params.stb_params.push_back(sizes);
            mem.unlock();
        }

        auto& kernel_selector = kernel_selector::space_to_batch_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(space_to_batch_params, space_to_batch_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto space_to_batch = new space_to_batch_gpu(arg, best_kernels[0]);

        return space_to_batch;
    }
};

namespace detail {

attach_space_to_batch_gpu::attach_space_to_batch_gpu() {
    auto val_fw = space_to_batch_gpu::create;
    implementation_map<space_to_batch>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<space_to_batch>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<space_to_batch>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<space_to_batch>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<space_to_batch>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<space_to_batch>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
