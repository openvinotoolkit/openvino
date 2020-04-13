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

#include "space_to_depth_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "space_to_depth/space_to_depth_kernel_selector.h"
#include "space_to_depth/space_to_depth_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {
struct space_to_depth_gpu : typed_primitive_gpu_impl<space_to_depth> {
    using parent = typed_primitive_gpu_impl<space_to_depth>;
    using parent::parent;

public:
    static primitive_impl* create(const space_to_depth_node& arg) {
        auto space_to_depth_params = get_default_params<kernel_selector::space_to_depth_params>(arg);
        auto space_to_depth_optional_params =
                get_default_optional_params<kernel_selector::space_to_depth_optional_params>(arg.get_program());

        space_to_depth_params.depth_mode = (arg.get_primitive()->mode == space_to_depth::blocks_first) ?
                                           kernel_selector::SpaceToDepthMode::BLOCKS_FIRST :
                                           kernel_selector::SpaceToDepthMode::DEPTH_FIRST;

        space_to_depth_params.block_size = arg.get_primitive()->block_size;

        auto& kernel_selector = kernel_selector::space_to_depth_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(space_to_depth_params, space_to_depth_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto space_to_depth = new space_to_depth_gpu(arg, best_kernels[0]);

        return space_to_depth;
    }
};

namespace detail {

attach_space_to_depth_gpu::attach_space_to_depth_gpu() {
    auto val_fw = space_to_depth_gpu::create;
    implementation_map<space_to_depth>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                            val_fw);
    implementation_map<space_to_depth>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                            val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
