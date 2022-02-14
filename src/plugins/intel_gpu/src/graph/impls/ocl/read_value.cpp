// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "read_value_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"

namespace cldnn {
namespace ocl {

struct read_value_impl : public typed_primitive_impl_ocl<read_value> {
    using parent = typed_primitive_impl_ocl<read_value>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<read_value_impl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, read_value_inst& instance) override {
        const auto arg = instance.argument;
        const auto variable_id = arg.variable_id;

        auto& variables = instance.get_network().get_variables();

        auto var_it = variables.find(variable_id);
        if (var_it == variables.end()) {
            CLDNN_ERROR_MESSAGE(instance.id(), "Variable " + variable_id + " not found");
        }

        const auto& variable = var_it->second;

        if (variable.memory->get_layout() != arg.output_layout) {
            CLDNN_ERROR_MESSAGE(instance.id(), "Layout mismatch");
        }

        std::vector<event::ptr> tmp_events(events);
        auto& stream = instance.get_network().get_stream();

        const auto ev_set_output = variable.is_set ? instance.output_memory().copy_from(stream, *variable.memory)
                                          : instance.output_memory().fill(stream, 0);

        tmp_events.push_back(ev_set_output);

        return parent::execute_impl(tmp_events, instance);
    }

public:
    static primitive_impl* create(read_value_node const& arg) { return new read_value_impl(arg, {}); }
};

namespace detail {

attach_read_value_impl::attach_read_value_impl() {
    implementation_map<read_value>::add(impl_types::ocl, read_value_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
