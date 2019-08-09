/*
// Copyright (c) 2019 Intel Corporation
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

#include "depth_to_space_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "depth_to_space/depth_to_space_kernel_selector.h"
#include "depth_to_space/depth_to_space_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {
struct depth_to_space_gpu : typed_primitive_gpu_impl<depth_to_space> {
    using parent = typed_primitive_gpu_impl<depth_to_space>;
    using parent::parent;

public:
    static primitive_impl* create(const depth_to_space_node& arg) {
        auto depth_to_space_params = get_default_params<kernel_selector::depth_to_space_params>(arg);
        auto depth_to_space_optional_params =
            get_default_optional_params<kernel_selector::depth_to_space_optional_params>(arg.get_program());

        depth_to_space_params.block_size = arg.get_primitive()->block_size;

        auto& kernel_selector = kernel_selector::depth_to_space_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(depth_to_space_params, depth_to_space_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto depth_to_space = new depth_to_space_gpu(arg, best_kernels[0]);

        return depth_to_space;
    }
};

namespace {
struct attach {
    attach() {
        auto val_fw = depth_to_space_gpu::create;
        implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                                val_fw);
        implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                                val_fw);
    }
    ~attach() = default;
};
attach attach_impl;
}  // namespace
}  // namespace gpu
}  // namespace cldnn
