// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "read_value_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace cpu {

struct read_value_impl : public typed_primitive_impl<read_value> {
    using parent = typed_primitive_impl<read_value>;
    using parent::parent;

    std::string variable_id;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::read_value_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<read_value_impl>(*this);
    }

    read_value_impl() : parent() {}

    explicit read_value_impl(const read_value_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<read_value>());
        const auto& node = arg.as<read_value>();
        variable_id = node.get_primitive()->variable_id;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << variable_id;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> variable_id;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, read_value_inst& instance) override {
        for (auto e : events) {
            e->wait();
        }

        auto& variable = instance.get_network().get_variable_memory(variable_id);

        if (variable.memory->get_layout() != instance.get_output_layout()) {
            CLDNN_ERROR_MESSAGE(instance.id(), "Layout mismatch");
        }

        if (!variable.is_set) {
            auto &stream = instance.get_network().get_stream();
            const auto ev_set_output = instance.output_memory().fill(stream, 0);
            return ev_set_output;
        }

        return instance.get_network().get_stream().create_user_event(true);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const read_value_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<read_value_impl>(arg);
    }
};

namespace detail {

attach_read_value_impl::attach_read_value_impl() {
    implementation_map<read_value>::add(impl_types::cpu, read_value_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::read_value_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::read_value)
