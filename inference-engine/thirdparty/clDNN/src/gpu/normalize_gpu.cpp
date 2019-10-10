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

#include "normalize_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "normalize/normalize_kernel_selector.h"
#include "normalize/normalize_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct normalize_gpu : typed_primitive_gpu_impl<normalize> {
    using parent = typed_primitive_gpu_impl<normalize>;
    using parent::parent;

protected:
     kernel::kernel_arguments_data get_arguments(typed_primitive_inst<normalize>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);
        args.scale_table = (memory_impl::cptr) &instance.scale_memory();
        return args;
    }

public:
    static primitive_impl* create(const normalize_node& arg) {
        auto norm_params = get_default_params<kernel_selector::normalize_params>(arg);
        auto norm_optional_params =
            get_default_optional_params<kernel_selector::normalize_optional_params>(arg.get_program());

        const auto& scale_layout = arg.scale().get_output_layout();

        norm_params.normMode = arg.get_primitive()->across_spatial ? kernel_selector::normalize_mode::ACROSS_SPATIAL
                                                                   : kernel_selector::normalize_mode::WITHIN_SPATIAL;
        norm_params.epsilon = arg.get_primitive()->epsilon;
        norm_params.scaleTable = convert_data_tensor(scale_layout).FlattenFeatureAndSpatials();

        auto& kernel_selector = kernel_selector::normalize_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(norm_params, norm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto lrn = new normalize_gpu(arg, best_kernels[0]);

        return lrn;
    }
};

namespace detail {

attach_normalize_gpu::attach_normalize_gpu() {
    implementation_map<normalize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                       normalize_gpu::create);
    implementation_map<normalize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                       normalize_gpu::create);
    implementation_map<normalize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                       normalize_gpu::create);
    implementation_map<normalize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                       normalize_gpu::create);
    implementation_map<normalize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                       normalize_gpu::create);
    implementation_map<normalize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                       normalize_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
