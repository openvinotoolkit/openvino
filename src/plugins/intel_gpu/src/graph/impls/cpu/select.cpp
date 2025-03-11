// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "select_inst.h"
#include "registry/implementation_map.hpp"

#include "openvino/op/select.hpp"

namespace cldnn {
namespace cpu {

struct select_impl : public typed_primitive_impl<select> {
    using parent = typed_primitive_impl<select>;
    using parent::parent;

    ov::op::AutoBroadcastSpec broadcast_spec;
    std::shared_ptr<ov::op::v1::Select> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::select_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<select_impl>(*this);
    }

    select_impl() : parent("select_cpu_impl") {}

    explicit select_impl(const select_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<select>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<select>();
        broadcast_spec = node.get_primitive()->broadcast_spec;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << make_data(&broadcast_spec, sizeof(ov::op::AutoBroadcastSpec));
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> make_data(&broadcast_spec, sizeof(ov::op::AutoBroadcastSpec));
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, select_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "select::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            stream.wait_for_events(events);
        }

        auto params = instance.get_impl_params();

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        if (!op) {
            op = std::make_shared<ov::op::v1::Select>();
            op->set_auto_broadcast(broadcast_spec);
        }

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::read_write> output_lock(output_mem_ptr, stream);
        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute select primitive with id ", instance.id());

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        if (pass_through_events) {
            return stream.group_events(events);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const select_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<select_impl>();
    }
};


namespace detail {

attach_select_impl::attach_select_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<select>::add(impl_types::cpu, shape_types::static_shape, select_impl::create, types, formats);
    implementation_map<select>::add(impl_types::cpu, shape_types::dynamic_shape, select_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::select_impl)
