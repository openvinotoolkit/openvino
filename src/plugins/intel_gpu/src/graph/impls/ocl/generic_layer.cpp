// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_layer_inst.h"
#include "intel_gpu/runtime/engine.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "register.hpp"
#include <vector>

namespace cldnn {
namespace ocl {

struct generic_layer_impl : typed_primitive_impl<generic_layer> {
    kernel_selector::cl_kernel_data _cl_kernel_data;
    std::vector<kernel::ptr> _kernels;
    kernel_id _kernel_id;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generic_layer_impl>(*this);
    }

    generic_layer_impl(const generic_layer_impl& other) : _cl_kernel_data(other._cl_kernel_data), _kernels({}) , _kernel_id(other._kernel_id) {
        if (other._kernels.empty()) {
            throw std::runtime_error("Can't copy generic_layer_impl node: kernels vector is empty");
        }
        _kernels.push_back(std::move(other._kernels.front()->clone()));
    }

    generic_layer_impl(const generic_layer_node& arg) : _cl_kernel_data(*arg.get_primitive()->generic_params.clKernel), _kernels({}) {
        _kernel_id = arg.get_program().add_kernel(_cl_kernel_data.code.kernelString);
    }

    void init_kernels(const program_node& node) override {
        _kernels.push_back(std::move(node.get_program().get_kernel(_kernel_id)));
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

static std::unique_ptr<primitive_impl> create(const generic_layer_node& arg) {
    return make_unique<generic_layer_impl>(arg);
}

namespace detail {
attach_generic_layer_impl::attach_generic_layer_impl() {
    implementation_map<generic_layer>::add(cldnn::impl_types::ocl, create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
