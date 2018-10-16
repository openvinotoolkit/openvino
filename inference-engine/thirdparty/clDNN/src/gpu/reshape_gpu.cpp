/*
// Copyright (c) 2017 Intel Corporation
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

#include "reshape_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "reshape/reshape_kernel_ref.h"
#include "reshape/reshape_kernel_selector.h"
#include "error_handler.h"

namespace cldnn { namespace gpu {

struct reshape_gpu : public typed_primitive_gpu_impl<reshape>
{
    using parent = typed_primitive_gpu_impl<reshape>;
    using parent::parent;

protected:

    virtual bool optimized_out(reshape_inst& instance) const override
    {
        return
            parent::optimized_out(instance) || _outer.is_in_place();
    }

public:

    static primitive_impl* create(reshape_node const& arg) 
    { 
        if (arg.is_in_place())
        {
            return new reshape_gpu(arg, {});
        }

        auto reorder_params = get_default_params<kernel_selector::reshape_params>(arg);
        auto reorder_optional_params = get_default_optional_params<kernel_selector::reshape_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::reshape_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(reorder_params, reorder_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto reshape = new reshape_gpu(arg, best_kernels[0]);

        return reshape;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<reshape>::add({
                { engine_types::ocl, reshape_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
