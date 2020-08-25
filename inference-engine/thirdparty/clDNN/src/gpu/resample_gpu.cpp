/*
// Copyright (c) 2016-2020 Intel Corporation
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

#include "resample_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_selector/core/actual_kernels/resample/resample_kernel_selector.h"
#include "kernel_selector/core/actual_kernels/resample/resample_kernel_base.h"

namespace cldnn {
namespace gpu {

namespace {
inline kernel_selector::sample_type convert_to_sample_type(resample_type type) {
    switch (type) {
        case resample_type::nearest:
            return kernel_selector::sample_type::NEAREST_NEIGHBOR;
        case resample_type::caffe_bilinear:
            return kernel_selector::sample_type::CAFFE_BILINEAR_INTERP;
        case resample_type::bilinear:
            return kernel_selector::sample_type::BILINEAR_INTERP;
        default:
            return kernel_selector::sample_type::NEAREST_NEIGHBOR;
    }
}
}  // namespace

struct resample_gpu : typed_primitive_gpu_impl<resample> {
    using parent = typed_primitive_gpu_impl<resample>;
    using parent::parent;

    static primitive_impl* create(const resample_node& arg) {
        auto us_params = get_default_params<kernel_selector::resample_params>(arg);
        auto us_optional_params =
            get_default_optional_params<kernel_selector::resample_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        if (primitive->with_activation)
            convert_activation_func_params(primitive, us_params.activations);

        us_params.resampleType = convert_to_sample_type(primitive->operation_type);

        if (primitive->operation_type == resample_type::bilinear) {
            us_params.pad_begin = primitive->pad_begin;
            us_params.pad_end = primitive->pad_end;
            us_params.align_corners = primitive->align_corners;
        }

        auto& kernel_selector = kernel_selector::resample_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(us_params, us_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto resample = new resample_gpu(arg, best_kernels[0]);

        return resample;
    }
};

namespace detail {

attach_resample_gpu::attach_resample_gpu() {
    implementation_map<resample>::add(
        {{std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f16, format::fs_b_yx_fsv32), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv16), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv16), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv4), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv4), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f32, format::byxf_af32), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::f16, format::byxf_af32), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::i8, format::byxf_af32), resample_gpu::create},
         {std::make_tuple(engine_types::ocl, data_types::u8, format::byxf_af32), resample_gpu::create}});
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
