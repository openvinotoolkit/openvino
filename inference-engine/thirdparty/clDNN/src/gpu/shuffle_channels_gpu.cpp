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

#include "shuffle_channels_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "shuffle_channels/shuffle_channels_kernel_selector.h"
#include "shuffle_channels/shuffle_channels_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct shuffle_channels_gpu : typed_primitive_gpu_impl<shuffle_channels> {
    using parent = typed_primitive_gpu_impl<shuffle_channels>;
    using parent::parent;

public:
    static primitive_impl* create(const shuffle_channels_node& arg) {
        auto shuffle_channels_params = get_default_params<kernel_selector::shuffle_channels_params>(arg);
        auto shuffle_channels_optional_params =
            get_default_optional_params<kernel_selector::shuffle_channels_optional_params>(arg.get_program());

        const int32_t number_of_dims = 4;
        int32_t axis = arg.get_primitive()->axis;

        if (axis < 0)
            axis += number_of_dims;

        shuffle_channels_params.group = arg.get_primitive()->group;
        shuffle_channels_params.axis = axis;

        auto& kernel_selector = kernel_selector::shuffle_channels_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(shuffle_channels_params, shuffle_channels_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto shuffle_channels = new shuffle_channels_gpu(arg, best_kernels[0]);

        return shuffle_channels;
    }
};

namespace detail {

attach_shuffle_channels_gpu::attach_shuffle_channels_gpu() {
    auto val_fw = shuffle_channels_gpu::create;
    implementation_map<shuffle_channels>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                              val_fw);
    implementation_map<shuffle_channels>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                              val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
