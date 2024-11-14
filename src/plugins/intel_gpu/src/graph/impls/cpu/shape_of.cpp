// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "shape_of_inst.h"
#include "impls/registry/implementation_map.hpp"

#include "openvino/op/shape_of.hpp"

namespace cldnn {
namespace cpu {

struct shape_of_impl : public typed_primitive_impl<shape_of> {
    using parent = typed_primitive_impl<shape_of>;
    using parent::parent;

    std::string variable_id;
    bool calculated = false;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::shape_of_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<shape_of_impl>(*this);
    }

    shape_of_impl() : parent("shape_of_cpu_impl") {}

    explicit shape_of_impl(const shape_of_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<shape_of>(), "[GPU] Incorrect program_node type");
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, shape_of_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "shape_of::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        auto output_mem_ptr = instance.output_memory_ptr();

        auto output_dt = instance.get_impl_params()->get_output_layout().data_type;

        if (output_dt == data_types::i32) {
            cldnn::mem_lock<int32_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);
            auto shape = instance.get_input_layout().get_shape();
            for (size_t i = 0; i < shape.size(); i++)
                output_lock[i] = static_cast<int32_t>(shape[i]);
        } else if (output_dt == data_types::i64) {
            cldnn::mem_lock<int64_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);
            auto shape = instance.get_input_layout().get_shape();
            for (size_t i = 0; i < shape.size(); i++)
                output_lock[i] = static_cast<int64_t>(shape[i]);
        } else {
            OPENVINO_THROW("[GPU] Couldn't execute shape_of operation: unsupported output data type (", output_dt , ")");
        }

        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }

        return stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const shape_of_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<shape_of_impl>();
    }
};


namespace detail {

attach_shape_of_impl::attach_shape_of_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
        data_types::i32,
        data_types::i64,
    };

    implementation_map<shape_of>::add(impl_types::cpu, shape_types::static_shape, shape_of_impl::create, types, formats);
    implementation_map<shape_of>::add(impl_types::cpu, shape_types::dynamic_shape, shape_of_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::shape_of_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::shape_of)
