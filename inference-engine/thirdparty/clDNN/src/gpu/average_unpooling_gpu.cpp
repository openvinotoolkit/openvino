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

#include "average_unpooling_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "average_unpooling/average_unpooling_kernel_selector.h"
#include "average_unpooling/average_unpooling_kernel_base.h"

namespace cldnn {
namespace gpu {

struct average_unpooling_gpu : typed_primitive_gpu_impl<average_unpooling> {
    using parent = typed_primitive_gpu_impl<average_unpooling>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<average_unpooling>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);
        return args;
    }

public:
    static primitive_impl* create(const average_unpooling_node& arg) {
        auto average_unpooling_params = get_default_params<kernel_selector::average_unpooling_params>(arg);
        auto average_unpooling_optional_params =
            get_default_optional_params<kernel_selector::average_unpooling_optional_params>(arg.get_program());
        auto& params = average_unpooling_params;

        auto primitive = arg.get_primitive();
        auto stride = primitive->stride;

        params.unpoolSize = {
            (uint32_t)primitive->size.spatial[0],
            (uint32_t)primitive->size.spatial[1],
        };

        params.unpoolStride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1]};

        auto& kernel_selector = kernel_selector::average_unpooling_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(average_unpooling_params, average_unpooling_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto average_unpool = new average_unpooling_gpu(arg, best_kernels[0]);

        return average_unpool;
    }
};

namespace detail {

attach_average_unpooling_gpu::attach_average_unpooling_gpu() {
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                                average_unpooling_gpu::create);
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                                average_unpooling_gpu::create);
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                                average_unpooling_gpu::create);
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                                average_unpooling_gpu::create);
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx),
                                                average_unpooling_gpu::create);
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb),
                                                average_unpooling_gpu::create);
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                                average_unpooling_gpu::create);
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                                average_unpooling_gpu::create);
    implementation_map<average_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf),
                                                average_unpooling_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
