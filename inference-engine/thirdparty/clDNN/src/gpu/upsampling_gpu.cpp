/*
// Copyright (c) 2016 Intel Corporation
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

#include "upsampling_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "upsampling/upsampling_kernel_selector.h"
#include "upsampling/upsampling_kernel_base.h"

namespace cldnn { namespace gpu {

namespace
{
    inline kernel_selector::sample_type convert_to_sample_type(upsampling_sample_type type)
    {
        switch (type)
        {
        case upsampling_sample_type::nearest:  return kernel_selector::sample_type::NEAREST;
        case upsampling_sample_type::bilinear:  return kernel_selector::sample_type::BILINEAR;
        default:
            return kernel_selector::sample_type::NEAREST;
        }
    }
}

struct upsampling_gpu : typed_primitive_gpu_impl<upsampling>
{
    using parent = typed_primitive_gpu_impl<upsampling>;
    using parent::parent;

    static primitive_impl* create(const upsampling_node& arg) 
    { 
        auto us_params = get_default_params<kernel_selector::upsampling_params>(arg);
        auto us_optional_params = get_default_optional_params<kernel_selector::upsampling_optional_params>(arg.get_program());
        
        const auto& primitive = arg.get_primitive();
        if(primitive->with_activation)
            convert_activation_func_params(primitive, us_params.activation);

        us_params.scale = primitive->scale;
        us_params.num_filter = primitive->num_filter;
        us_params.sampleType = convert_to_sample_type(primitive->sample_type);

        auto& kernel_selector = kernel_selector::upsampling_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(us_params, us_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto upsampling = new upsampling_gpu(arg, best_kernels[0]);

        return upsampling;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<upsampling>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), upsampling_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), upsampling_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), upsampling_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), upsampling_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), upsampling_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), upsampling_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
