/*
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
*/

#include "region_yolo_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "region_yolo/region_yolo_kernel_selector.h"
#include "region_yolo/region_yolo_kernel_ref.h"
#include "error_handler.h"

namespace cldnn { namespace gpu {

struct region_yolo_gpu : typed_primitive_gpu_impl<region_yolo>
{
    using parent = typed_primitive_gpu_impl<region_yolo>;
    using parent::parent;

    static primitive_impl* create(const region_yolo_node& arg)
    {
        auto ry_params = get_default_params<kernel_selector::region_yolo_params>(arg);
        auto ry_optional_params = get_default_optional_params<kernel_selector::region_yolo_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        ry_params.coords = primitive->coords;
        ry_params.classes = primitive->classes;
        ry_params.num = primitive->num;
        ry_params.do_softmax = primitive->do_softmax;
        ry_params.mask_size = primitive->mask_size;

        auto& kernel_selector = kernel_selector::region_yolo_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ry_params, ry_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto region_yolo_node = new region_yolo_gpu(arg, best_kernels[0]);

        return region_yolo_node;
    };
};

namespace {
    struct attach {
        attach() {
            implementation_map<region_yolo>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), region_yolo_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), region_yolo_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), region_yolo_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), region_yolo_gpu::create }
             });
        }
        ~attach() {}
    };

    attach attach_impl;
}

} // namespace gpu
} // namespace cldnn
