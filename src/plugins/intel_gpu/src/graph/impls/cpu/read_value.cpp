// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "read_value_inst.h"
#include "impls/implementation_map.hpp"
#include "register.hpp"

namespace cldnn {
namespace cpu {

struct read_value_impl : public typed_primitive_impl<read_value> {
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<read_value_impl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, read_value_inst& instance) override {
        for (auto e : events) {
            e->wait();
        }
        const auto arg = instance.argument;
        const auto variable_id = arg.variable_id;

        auto& variable = instance.get_network().get_variable_memory(variable_id);

        if (variable.memory->get_layout() != arg.output_layout) {
            CLDNN_ERROR_MESSAGE(instance.id(), "Layout mismatch");
        }

        if (!variable.is_set) {
            auto &stream = instance.get_network().get_stream();
            const auto ev_set_output = instance.output_memory().fill(stream, 0);
            return ev_set_output;
        }

        return instance.get_network().get_stream().create_user_event(true);
    }

    void init_kernels() override {}

public:
    static primitive_impl* create(read_value_node const& arg) { return new read_value_impl{}; }
};

namespace detail {

attach_read_value_impl::attach_read_value_impl() {
    implementation_map<read_value>::add(impl_types::cpu, read_value_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn
