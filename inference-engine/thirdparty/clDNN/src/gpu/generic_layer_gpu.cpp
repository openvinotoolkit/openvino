// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_layer_inst.h"
#include "cldnn/runtime/engine.hpp"
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
    std::vector<kernel::ptr> _kernels;
    kernel_id _kernel_id;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generic_layer_gpu>(*this);
    }

    generic_layer_gpu(const generic_layer_gpu& other)
    : outer(other.outer)
    , _cl_kernel_data(other._cl_kernel_data)
    , _kernels({})
    , _kernel_id(other._kernel_id) {
        if (other._kernels.empty()) {
            throw std::runtime_error("Can't copy generic_layer_gpu node: kernels vector is empty");
        }
        _kernels.push_back(other._kernels.front()->clone());
    }

    generic_layer_gpu(const generic_layer_node& arg)
        : outer(arg)
        , _cl_kernel_data(*outer.get_primitive()->generic_params.clKernel.get())
        , _kernels() {
        _kernel_id = outer.get_program().add_kernel(outer.get_primitive()->generic_params.clKernel->code.kernelString);
    }

    void init_kernels() override {
        _kernels.push_back(outer.get_program().get_kernel(_kernel_id));
    }

    void set_arguments_impl(generic_layer_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        args.scalars = &_cl_kernel_data.params.scalars;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }
        args.output = instance.output_memory_ptr();
        stream.set_arguments(*_kernels.front(), _cl_kernel_data.params, args);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, generic_layer_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        args.scalars = &_cl_kernel_data.params.scalars;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }
        args.output = instance.output_memory_ptr();
        return stream.enqueue_kernel(*_kernels.front(), _cl_kernel_data.params, args, events, true);
    }
};

// TODO: move this file to cpu folder and add a new traget to 'cldnn::engine_types'
struct generic_layer_cpu : typed_primitive_impl<generic_layer> {
    const generic_layer_node& outer;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generic_layer_cpu>(*this);
    }

    explicit generic_layer_cpu(const generic_layer_node& arg) : outer(arg) {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, generic_layer_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        auto input_mem = instance.input_memory_ptr();
        auto output_mem = instance.output_memory_ptr();

        auto ev = stream.create_user_event(false);
        std::vector<event::ptr> tmp_events(events);

        for (auto& a : events) {
            a->wait();
        }

        mem_lock<uint8_t> old_pointer(input_mem, stream);
        mem_lock<uint8_t> new_pointer(output_mem, stream);

        const auto& cpu_kernel = *outer.get_primitive()->generic_params.cpuKernel.get();

        cpu_kernel.Execute(old_pointer.data(), old_pointer.size(), new_pointer.data(), new_pointer.size());

        ev->set();
        return ev;
    }

    void init_kernels() override {}
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
