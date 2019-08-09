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

#include "lrn_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "lrn/lrn_kernel_selector.h"
#include "lrn/lrn_kernel_base.h"

namespace cldnn {
namespace gpu {

struct lrn_gpu : typed_primitive_gpu_impl<lrn> {
    using parent = typed_primitive_gpu_impl<lrn>;
    using parent::parent;

    static primitive_impl* create(const lrn_node& arg) {
        auto lrn_params = get_default_params<kernel_selector::lrn_params>(arg);
        auto lrn_optional_params = get_default_optional_params<kernel_selector::lrn_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();

        lrn_params.alpha = primitive->alpha;
        lrn_params.beta = primitive->beta;
        lrn_params.k = primitive->k;
        lrn_params.localSize = primitive->size;
        lrn_params.divMode = kernel_selector::kernel_divider_mode::FIXED;
        lrn_params.normMode = primitive->norm_region == cldnn_lrn_norm_region_within_channel
                                  ? kernel_selector::lrn_mode::WITHIN_CHANNEL
                                  : kernel_selector::lrn_mode::ACROSS_CHANNEL;

        auto& kernel_selector = kernel_selector::lrn_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(lrn_params, lrn_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto lrn = new lrn_gpu(arg, best_kernels[0]);

        return lrn;
    }
};

namespace {
struct attach {
    attach() {
        implementation_map<lrn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                     lrn_gpu::create);
        implementation_map<lrn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                     lrn_gpu::create);
        implementation_map<lrn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                     lrn_gpu::create);
        implementation_map<lrn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                     lrn_gpu::create);
        implementation_map<lrn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                     lrn_gpu::create);
        implementation_map<lrn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                     lrn_gpu::create);
    }
    ~attach() {}
};
attach attach_impl;
}  // namespace
}  // namespace gpu
}  // namespace cldnn
