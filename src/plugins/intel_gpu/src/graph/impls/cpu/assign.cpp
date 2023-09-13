// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "assign_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace cpu {

struct assign_impl : public typed_primitive_impl<assign> {
    using parent = typed_primitive_impl<assign>;
    using parent::parent;

    std::string variable_id;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::assign_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<assign_impl>(*this);
    }

    assign_impl() : parent() {}

    explicit assign_impl(const assign_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<assign>());
        const auto& node = arg.as<assign>();
        variable_id = node.get_primitive()->variable_id;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << variable_id;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> variable_id;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, assign_inst& instance) override {
        auto& variable = instance.get_network().get_variable_memory(variable_id);

        if (variable.memory->get_layout() != instance.get_output_layout()) {
            CLDNN_ERROR_MESSAGE(instance.id(), "Layout mismatch");
        }

        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        const auto ev_set_memory = variable.memory->copy_from(stream, instance.input_memory());
        variable.is_set = true;

        return ev_set_memory;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const assign_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<assign_impl>();
    }
};


namespace detail {

attach_assign_impl::attach_assign_impl() {
    implementation_map<assign>::add(impl_types::cpu, assign_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::assign_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::assign)
