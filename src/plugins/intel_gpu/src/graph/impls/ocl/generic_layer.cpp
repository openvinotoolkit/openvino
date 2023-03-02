// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "generic_layer_inst.h"

namespace cldnn {
namespace ocl {

struct generic_layer_impl : typed_primitive_impl<generic_layer> {
    using parent = typed_primitive_impl<generic_layer>;
    using parent::parent;

    kernel_selector::cl_kernel_data _cl_kernel_data;
    std::vector<kernel::ptr> _kernels;
    kernel_id _kernel_id;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generic_layer_impl>(*this);
    }

    generic_layer_impl() : parent() {}

    generic_layer_impl(const generic_layer_impl& other)
    : _cl_kernel_data(other._cl_kernel_data)
    , _kernels({})
    , _kernel_id(other._kernel_id) {
        if (other._kernels.empty()) {
            throw std::runtime_error("Can't copy generic_layer_impl node: kernels vector is empty");
        }
        _kernels.push_back(other._kernels.front()->clone());
    }

    generic_layer_impl(const generic_layer_node& arg)
        : _cl_kernel_data(*arg.get_primitive()->generic_params.clKernel.get())
        , _kernels() {
        _kernel_id = arg.get_program().add_kernel(arg.get_primitive()->generic_params.clKernel->code.kernelString);
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob <<_cl_kernel_data;
        ob << _kernel_id;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> _cl_kernel_data;
        ib >> _kernel_id;
    }

    void init_kernels(const kernels_cache& kernels_cache) override {
        _kernels.push_back(kernels_cache.get_kernel(_kernel_id));
    }

    void set_arguments_impl(generic_layer_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        args.scalars = &_cl_kernel_data.params.scalars;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }
        args.outputs.push_back(instance.output_memory_ptr());
        stream.set_arguments(*_kernels.front(), _cl_kernel_data.params, args);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, generic_layer_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        args.scalars = &_cl_kernel_data.params.scalars;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }
        args.outputs.push_back(instance.output_memory_ptr());
        return stream.enqueue_kernel(*_kernels.front(), _cl_kernel_data.params, args, events, true);
    }
};

// TODO: move this file to cpu folder and add a new traget to 'cldnn::engine_types'
struct generic_layer_cpu : typed_primitive_impl<generic_layer> {
    const generic_layer_node& outer;
    DECLARE_OBJECT_TYPE_SERIALIZATION

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

        mem_lock<uint8_t, mem_lock_type::read> old_pointer(input_mem, stream);
        mem_lock<uint8_t, mem_lock_type::write> new_pointer(output_mem, stream);

        const auto& cpu_kernel = *outer.get_primitive()->generic_params.cpuKernel.get();

        cpu_kernel.Execute(old_pointer.data(), old_pointer.size(), new_pointer.data(), new_pointer.size());

        ev->set();
        return ev;
    }

    void init_kernels(const kernels_cache&) override {}
};

static std::unique_ptr<primitive_impl> create(const generic_layer_node& arg, const kernel_impl_params&) {
    if (arg.get_primitive()->generic_params.engine == kernel_selector::generic_kernel_params::Engine::GPU) {
        return make_unique<generic_layer_impl>(arg);
    } else {
        return make_unique<generic_layer_cpu>(arg);
    }
}

namespace detail {
attach_generic_layer_impl::attach_generic_layer_impl() {
    implementation_map<generic_layer>::add(cldnn::impl_types::ocl, create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::generic_layer_impl)
ASSIGN_TYPE_NAME(cldnn::ocl::generic_layer_cpu)
