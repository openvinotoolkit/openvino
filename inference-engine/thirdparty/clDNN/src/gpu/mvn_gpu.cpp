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

#include "mvn_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "mvn/mvn_kernel_selector.h"
#include "mvn/mvn_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct mvn_gpu : typed_primitive_gpu_impl<mvn> {
    using parent = typed_primitive_gpu_impl<mvn>;
    using parent::parent;

public:
    static primitive_impl* create(const mvn_node& arg) {
        auto mvn_params = get_default_params<kernel_selector::mvn_params>(arg);
        auto mvn_optional_params = get_default_optional_params<kernel_selector::mvn_optional_params>(arg.get_program());

        mvn_params.mvnMode = arg.get_primitive()->across_channels ? kernel_selector::mvn_mode::ACROSS_CHANNELS
                                                                  : kernel_selector::mvn_mode::WITHIN_CHANNELS;
        mvn_params.mvnNormalizeVariance = arg.get_primitive()->normalize_variance;
        mvn_params.epsilon = arg.get_primitive()->epsilon;

        auto& kernel_selector = kernel_selector::mvn_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(mvn_params, mvn_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto mvn = new mvn_gpu(arg, best_kernels[0]);

        return mvn;
    }
};

namespace detail {

attach_mvn_gpu::attach_mvn_gpu() {
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                 mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                 mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                 mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                 mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                 mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                 mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx),
                                 mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx),
                                 mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx_f16),
        mvn_gpu::create);
    implementation_map<mvn>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx_f16),
        mvn_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
