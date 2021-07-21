// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_primitive_inst.h"
#include "cldnn/runtime/engine.hpp"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "network_impl.h"
#include "jitter.h"
#include "cldnn/runtime/error_handler.hpp"
#include "register_gpu.hpp"

#include <map>
#include <sstream>
#include <vector>
#include <memory>
#include <string>

using namespace cldnn;
namespace kernel_selector {
    using jit_constants = kernel_selector::JitConstants;
}

namespace neural {

struct generic_primitive_gpu : typed_primitive_impl<generic_primitive> {
    const generic_primitive_node& outer;
    const generic_primitive::execute_function callback_function;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generic_primitive_gpu>(*this);
    }

    generic_primitive_gpu(const generic_primitive_gpu& other)
            : outer(other.outer)
            , callback_function(other.callback_function) {
    }

    generic_primitive_gpu(const generic_primitive_node& arg,
                          const generic_primitive::execute_function& impl)
            : outer(arg)
            , callback_function(impl) {
    }

    void init_kernels() override { }

    void set_arguments_impl(generic_primitive_inst& instance) override {
        // FIXME: not used?
        // auto& stream = instance.get_network().get_stream();
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            generic_primitive_inst& instance) override {
        //auto& stream = instance.get_network().get_stream();
        std::vector<memory::ptr> inputs;
        inputs.reserve(instance.inputs_memory_count());
        for (auto& dep : instance.dependencies()) {
            inputs.push_back(dep->output_memory_ptr());
        }
        // TODO: support multiple outputs?
        std::vector<memory::ptr> outputs;
        outputs.push_back(instance.output_memory_ptr());

        return instance.node.get_primitive()->callback_function(
            events, inputs, outputs);
    }
};

static primitive_impl* create(const generic_primitive_node& arg) {
    const auto primitive = arg.get_primitive().get();
    return new generic_primitive_gpu(arg, primitive->callback_function);
}

}  // namespace neural

namespace cldnn { namespace gpu { namespace detail {

attach_generic_primitive_gpu::attach_generic_primitive_gpu() {
    implementation_map<generic_primitive>::add({{cldnn::engine_types::ocl, neural::create}});
}

} } }  // namespace cldnn::gpu::detail
