// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "assign_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"

namespace cldnn {
namespace ocl {

struct assign_impl : public typed_primitive_impl_ocl<assign> {
    using parent = typed_primitive_impl_ocl<assign>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<assign_impl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, assign_inst& instance) override {
        const auto arg = instance.argument;
        const auto variable_id = arg.variable_id;

        auto& variables = instance.get_network().get_variables();

        auto var_it = variables.find(variable_id);
        if (var_it == variables.end()) {
            CLDNN_ERROR_MESSAGE(instance.id(), "Variable " + variable_id + " not found");
        }

        auto& variable = var_it->second;

        if (variable.memory->get_layout() != arg.output_layout) {
            CLDNN_ERROR_MESSAGE(instance.id(), "Layout mismatch");
        }

        std::vector<event::ptr> tmp_events(events);
        auto& stream = instance.get_network().get_stream();

        stream.enqueue_barrier(); // input must be ready before proceed

        const auto ev_set_memory = variable.memory->copy_from(stream, instance.input_memory());
        tmp_events.push_back(ev_set_memory);
        variable.is_set = true;

        const auto ev_set_output = instance.output_memory().copy_from(stream, *variable.memory);
        tmp_events.push_back(ev_set_output);

        return parent::execute_impl(tmp_events, instance);
    }

public:
    static primitive_impl* create(assign_node const& arg) { return new assign_impl(arg, {}); }
};


namespace detail {

attach_assign_impl::attach_assign_impl() {
    implementation_map<assign>::add(impl_types::ocl, assign_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
