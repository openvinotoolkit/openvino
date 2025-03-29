// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "assign_inst.h"
#include "registry/implementation_map.hpp"
#include "register.hpp"

namespace cldnn {
namespace cpu {

struct assign_impl : public typed_primitive_impl<assign> {
    using parent = typed_primitive_impl<assign>;
    using parent::parent;

    std::string variable_id;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::assign_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<assign_impl>(*this);
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
        parent::save(ob);
        ob << variable_id;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> variable_id;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, assign_inst& instance) override {
        auto& variable = instance.get_network().get_variable(variable_id);

        OPENVINO_ASSERT(variable.get_layout() == instance.get_output_layout(),
                        "[GPU] Layout mismatch: variable layout: ", variable.get_layout().to_short_string(),
                        " assign output layout: ", instance.get_output_layout().to_short_string());

        auto& stream = instance.get_network().get_stream();

        stream.wait_for_events(events);

        const auto ev_set_memory = variable.get_memory()->copy_from(stream, instance.input_memory(), 0, 0, variable.get_layout().bytes_count(), true);
        variable.set();

        return ev_set_memory;
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const assign_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<assign_impl>();
    }
};


namespace detail {

attach_assign_impl::attach_assign_impl() {
    implementation_map<assign>::add(impl_types::cpu, shape_types::dynamic_shape, assign_impl::create, {});
    implementation_map<assign>::add(impl_types::cpu, shape_types::static_shape, assign_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::assign_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::assign)
