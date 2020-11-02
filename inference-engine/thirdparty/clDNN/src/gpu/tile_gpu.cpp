/*
// Copyright (c) 2018-2020 Intel Corporation
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

#include "tile_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "tile/tile_kernel_selector.h"
#include "tile/tile_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct tile_gpu : typed_primitive_gpu_impl<tile> {
    using parent = typed_primitive_gpu_impl<tile>;
    using parent::parent;

public:
    static primitive_impl* create(const tile_node& arg) {
        auto tile_params = get_default_params<kernel_selector::tile_params>(arg);
        auto tile_optional_params =
            get_default_optional_params<kernel_selector::tile_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::tile_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(tile_params, tile_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto tile = new tile_gpu(arg, best_kernels[0]);

        return tile;
    }
};

namespace detail {

attach_tile_gpu::attach_tile_gpu() {
    auto val_fw = tile_gpu::create;

    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);

    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);

    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfwzyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfwzyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<tile>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
