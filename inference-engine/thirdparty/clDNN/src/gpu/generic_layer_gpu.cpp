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

#include "generic_layer_inst.h"
#include "runtime/kernel.h"
#include "runtime/engine_impl.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "network_impl.h"
#include "register_gpu.hpp"
#include <vector>

using namespace cldnn;

namespace neural {

struct generic_layer_gpu : typed_primitive_impl<generic_layer> {
    const generic_layer_node& outer;
    const kernel_selector::cl_kernel_data& _cl_kernel_data;
    std::vector<gpu::kernel> _kernels;
    gpu::kernel_id _kernel_id;

    generic_layer_gpu(const generic_layer_node& arg)
        : outer(arg)
        , _cl_kernel_data(*outer.get_primitive()->generic_params.clKernel.get())
        , _kernels() {
        _kernel_id = outer.get_program().add_kernel(outer.get_primitive()->generic_params.clKernel->code.kernelString);
    }

    void init_kernels_impl(generic_layer_inst& instance) override {
        auto context = instance.get_network().get_engine().get_context();

        gpu::kernel kernel(context, outer.get_program().get_kernel(_kernel_id, false), _kernel_id);
        _kernels.emplace_back(std::move(kernel));
    }

    void set_arguments_impl(generic_layer_inst& instance) override {
        auto net_id = instance.get_network().get_id();
        gpu::kernel_arguments_data args;
        args.scalars = &_cl_kernel_data.params.scalars;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back((memory_impl::cptr) &instance.input_memory(i));
        }
        args.output = (memory_impl::cptr) &instance.output_memory();
        _kernels.front().set_arguments(net_id, _cl_kernel_data.params, args);
    }

    void cleanup_impl(generic_layer_inst& instance) override {
        auto net_id = instance.get_network().get_id();
        _kernels.front().cleanup(net_id);
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, generic_layer_inst& instance) override {
        uint32_t net_id = instance.get_network().get_id();
        _kernels.front().set_output_event(net_id, instance.node.is_output());
        return _kernels.front().run(net_id, _cl_kernel_data.params, events);
    }
};

// TODO: move this file to cpu folder and add a new traget to 'cldnn::engine_types'
struct generic_layer_cpu : typed_primitive_impl<generic_layer> {
    const generic_layer_node& outer;

    explicit generic_layer_cpu(const generic_layer_node& arg) : outer(arg) {}

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, generic_layer_inst& instance) override {
        uint32_t net_id = instance.get_network().get_id();
        auto& input_mem = instance.input_memory();
        auto& output_mem = instance.output_memory();

        std::vector<event_impl::ptr> tmp_events(events);

        for (auto& a : events) {
            a->wait();
        }

        mem_lock<uint8_t> old_pointer(input_mem);
        mem_lock<uint8_t> new_pointer(output_mem);

        const auto& cpu_kernel = *outer.get_primitive()->generic_params.cpuKernel.get();

        cpu_kernel.Execute(old_pointer.data(), old_pointer.size(), new_pointer.data(), new_pointer.size());

        return instance.get_network().get_engine().create_user_event(net_id, true);
    }
};

static primitive_impl* create(const generic_layer_node& arg) {
    if (arg.get_primitive()->generic_params.engine == kernel_selector::generic_kernel_params::Engine::GPU) {
        return new generic_layer_gpu(arg);
    } else {
        return new generic_layer_cpu(arg);
    }
}

}  // namespace neural

namespace cldnn { namespace gpu { namespace detail {
    attach_generic_layer_gpu::attach_generic_layer_gpu() {
        implementation_map<generic_layer>::add({ {cldnn::engine_types::ocl, neural::create} });
    }

} } }  // namespace cldnn::gpu::detail
