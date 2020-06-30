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
#include "kernels_cache.h"
#include "event_impl.h"

#include "kernel_selector_helper.h"
#include <memory>
#include <vector>

namespace cldnn {
namespace gpu {

class kernel : public context_holder {
    uint32_t _prog_id;
    kernels_cache::kernel_id _kernel_id;
    bool _one_time_kernel;  // If this flag is true, the kernel is intended to be executed only once (can be removed
                            // later from the cache).

public:
    explicit kernel(std::shared_ptr<gpu_toolkit> context,
                    const std::shared_ptr<kernel_selector::kernel_string>& kernel_string,
                    uint32_t prog_id,
                    bool dump_custom_program = false,
                    bool one_time_kernel = false)
        : context_holder(context),
          _prog_id(prog_id),
          _kernel_id(
              context->get_kernels_cache(prog_id).set_kernel_source(kernel_string, dump_custom_program, one_time_kernel)),
          _one_time_kernel(one_time_kernel) {}

    kernel(const kernel& other)
        : context_holder(other.context()), _prog_id(other._prog_id), _kernel_id(other._kernel_id), _one_time_kernel(other._one_time_kernel) {}

    kernel& operator=(const kernel& other) {
        if (this == &other) {
            return *this;
        }

        _kernel_id = other._kernel_id;
        _prog_id = other._prog_id;
        _one_time_kernel = other._one_time_kernel;

        return *this;
    }

    struct kernel_arguments_data {
        std::vector<memory_impl::cptr> inputs;
        std::vector<memory_impl::cptr> intermediates;
        memory_impl::cptr output;
        memory_impl::cptr weights;
        memory_impl::cptr recurrent;
        memory_impl::cptr hidden;
        memory_impl::cptr cell;
        memory_impl::cptr bias;
        memory_impl::cptr weights_zero_points;
        memory_impl::cptr activations_zero_points;
        memory_impl::cptr compensation;
        memory_impl::cptr lookup_table;
        memory_impl::cptr scale_table;
        memory_impl::cptr slope;
        // used for fused primitives
        std::vector<memory_impl::cptr> fused_op_inputs;
        int32_t split = 0;
        float lr;
        const kernel_selector::kernel_scalar_arguments* scalars = nullptr;
    };

    void set_output_event(uint32_t net_id, bool is_out_event) {
        context()->set_output_event(net_id, is_out_event);
    }

    event_impl::ptr run(uint32_t queue_id,
                        const kernel_selector::cl_kernel_data& kernel_data,
                        const std::vector<event_impl::ptr>& dependencies,
                        const kernel_arguments_data& args) const;
};

}  // namespace gpu
}  // namespace cldnn
