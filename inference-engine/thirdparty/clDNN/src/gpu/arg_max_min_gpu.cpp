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

#include "arg_max_min_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "arg_max_min/arg_max_min_kernel_selector.h"
#include "arg_max_min/arg_max_min_kernel_base.h"
#include "kernel_runner.h"

namespace cldnn {
namespace gpu {

struct arg_max_min_gpu : typed_primitive_gpu_impl<arg_max_min> {
    using parent = typed_primitive_gpu_impl<arg_max_min>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<arg_max_min>& instance,
                                                        int32_t) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, 0);

        if (args.inputs.size() == 3) {
            args.inputs.erase(args.inputs.begin() + 1);  // erase constant input in case of TOP_K
        }

        return args;
    }

public:
    static primitive_impl* create(const arg_max_min_node& arg) {
        const auto& primitive = arg.get_primitive();

        const auto& axis = primitive->axis;
        const auto& top_k = primitive->top_k;
        const auto& out_type = primitive->output_type;
        const auto& sort_type = primitive->sort;
        const auto& with_axis = primitive->with_axis;
        const auto& values_first = primitive->values_first;
        const auto& outputs_num = primitive->input.size() == 3 ? 2 : 1;  // second output passed as input for TOP_K layer

        auto argm_params = get_default_params<kernel_selector::arg_max_min_params>(arg);
        auto argm_optional_params =
            get_default_optional_params<kernel_selector::arg_max_min_optional_params>(arg.get_program());

        argm_params.outputs_num = outputs_num;
        argm_params.topK = top_k;
        if (with_axis) {
            switch (axis) {
                case arg_max_min::batch:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::BATCH;
                    break;
                case arg_max_min::feature:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::FEATURE;
                    break;
                case arg_max_min::x:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::X;
                    break;
                case arg_max_min::y:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::Y;
                    break;
                case arg_max_min::z:
                    argm_params.argMaxMinAxis = kernel_selector::argm_axis::Z;
                    break;
                default:
                    break;
            }
        }

        if (out_type == primitive->max)
            argm_params.argMaxMinOut = kernel_selector::argm_output::MAX;
        else
            argm_params.argMaxMinOut = kernel_selector::argm_output::MIN;

        if (sort_type == primitive->sort_by_values)
            argm_params.argMaxMinSortType = kernel_selector::argm_sort::VALUE;
        else
            argm_params.argMaxMinSortType = kernel_selector::argm_sort::INDEX;

        if (outputs_num == 2) {
            argm_params.inputs.push_back(convert_data_tensor(arg.get_dependency(2).get_output_layout()));
        }

        argm_params.values_first = values_first;

        auto& kernel_selector = kernel_selector::arg_max_min_kernel_selector::Instance();

        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(argm_params, argm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto conv = new arg_max_min_gpu(arg, best_kernels[0]);

        return conv;
    }
};

namespace {
struct attach {
    attach() {
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                             arg_max_min_gpu::create);
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                             arg_max_min_gpu::create);
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx),
                                             arg_max_min_gpu::create);
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx),
                                             arg_max_min_gpu::create);
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx),
                                             arg_max_min_gpu::create);
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx),
                                             arg_max_min_gpu::create);
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                             arg_max_min_gpu::create);
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                             arg_max_min_gpu::create);
        implementation_map<arg_max_min>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb),
                                             arg_max_min_gpu::create);
    }
    ~attach() {}
};
attach attach_impl;
}  // namespace
}  // namespace gpu
}  // namespace cldnn
