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

#include "extract_image_patches_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"

#include "extract_image_patches/extract_image_patches_kernel_selector.h"
#include "extract_image_patches/extract_image_patches_kernel_ref.h"

namespace cldnn {
namespace gpu {

struct extract_image_patches_gpu : typed_primitive_gpu_impl<extract_image_patches> {
    using parent = typed_primitive_gpu_impl<extract_image_patches>;
    using parent::parent;

public:
    static primitive_impl* create(const extract_image_patches_node& arg) {
        auto params = get_default_params<kernel_selector::extract_image_patches_params>(arg);
        auto optional_params =
            get_default_optional_params<kernel_selector::extract_image_patches_optional_params>(arg.get_program());

        params.sizes = arg.get_primitive()->sizes;
        params.strides = arg.get_primitive()->strides;
        params.rates = arg.get_primitive()->rates;
        params.auto_pad = arg.get_primitive()->auto_pad;

        auto& kernel_selector = kernel_selector::extract_image_patches_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto extract_image_patches = new extract_image_patches_gpu(arg, best_kernels[0]);

        return extract_image_patches;
    }
};

namespace detail {

attach_extract_image_patches_gpu::attach_extract_image_patches_gpu() {
    implementation_map<extract_image_patches>::add(
        {{std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), extract_image_patches_gpu::create}});
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
