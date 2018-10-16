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

#include "permute_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "permute/permute_kernel_selector.h"
#include "permute/permute_kernel_ref.h"

using namespace cldnn;

namespace cldnn { namespace gpu {

struct permute_gpu : typed_primitive_gpu_impl<permute>
{
    using parent = typed_primitive_gpu_impl<permute>;
    using parent::parent;

    static primitive_impl* create(const permute_node& arg)
    {
        auto permute_params = get_default_params<kernel_selector::permute_params>(arg);
        auto permute_optional_params = get_default_optional_params<kernel_selector::permute_optional_params>(arg.get_program());

        uint16_t max_input_index = (uint16_t)(permute_params.inputs[0].GetDims().size() - 1);
        const auto& permute_order = arg.get_primitive()->permute_order;
        for (size_t i = 0; i < permute_order.size(); i++)
        {
            auto order = permute_order[permute_order.size() - 1 - i];
            permute_params.order.push_back(max_input_index - order);
        }
        auto& kernel_selector = kernel_selector::permute_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(permute_params, permute_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto permute = new permute_gpu(arg, best_kernels[0]);

        return permute;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<permute>::add({
                { engine_types::ocl, permute_gpu::create },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }