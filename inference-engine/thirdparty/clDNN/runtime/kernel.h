/*
// Copyright (c) 2016-2019 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "ocl_toolkit.h"
#include "memory_impl.h"
#include "ocl_common.h"
#include "event_impl.h"
#include "kernel_args.h"

#include <memory>
#include <vector>

namespace cldnn {
namespace gpu {

class kernel : public context_holder {
    kernel_type _compiled_kernel;
    std::string _kernel_id;
    std::map<uint32_t, kernel_type> _cl_kernels;

public:
    explicit kernel(std::shared_ptr<gpu_toolkit> context,
                    const kernel_type& compiled_kernel,
                    const std::string& kernel_id)
        : context_holder(context)
        , _compiled_kernel(compiled_kernel)
        , _kernel_id(kernel_id)
        , _cl_kernels({}) {}

    kernel(const kernel& other)
        : context_holder(other.context())
        , _compiled_kernel(other._compiled_kernel)
        , _kernel_id(other._kernel_id)
        , _cl_kernels(other._cl_kernels) {}

    kernel& operator=(const kernel& other) {
        if (this == &other) {
            return *this;
        }

        _kernel_id = other._kernel_id;
        _compiled_kernel = other._compiled_kernel;
        _cl_kernels = other._cl_kernels;

        return *this;
    }

    void set_output_event(uint32_t net_id, bool is_out_event) {
        context()->set_output_event(net_id, is_out_event);
    }

    void cleanup(uint32_t queue_id);
    void set_arguments(uint32_t queue_id,
                       const kernel_arguments_desc& args_desc,
                       const kernel_arguments_data& args);
    event_impl::ptr run(uint32_t queue_id,
                        const kernel_arguments_desc& args_desc,
                        const std::vector<event_impl::ptr>& dependencies) const;
};

}  // namespace gpu
}  // namespace cldnn
