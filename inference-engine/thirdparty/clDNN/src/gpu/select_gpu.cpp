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

#include "select_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "select/select_kernel_selector.h"
#include "select/select_kernel_base.h"

namespace cldnn {
namespace gpu {

struct select_gpu : typed_primitive_gpu_impl<select> {
    using parent = typed_primitive_gpu_impl<select>;
    using parent::parent;

public:
    static primitive_impl* create(const select_node& arg) {
        auto select_params = get_default_params<kernel_selector::select_params>(arg);
        auto select_optional_params =
            get_default_optional_params<kernel_selector::select_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            select_params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }

        auto& kernel_selector = kernel_selector::select_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(select_params, select_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto select = new select_gpu(arg, best_kernels[0]);

        return select;
    }
};

namespace {
struct attach {
    attach() {
        implementation_map<select>::add(
            {{std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::u8, format::yxfb), select_gpu::create},

             {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), select_gpu::create},

             {std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), select_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::u8, format::byxf), select_gpu::create}});
    }

    ~attach() {}
};
attach attach_impl;
}  // namespace
}  // namespace gpu
}  // namespace cldnn
