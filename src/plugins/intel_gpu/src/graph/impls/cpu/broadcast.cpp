// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "broadcast_inst.h"
#include "impls/registry/implementation_map.hpp"

#include "openvino/op/broadcast.hpp"

namespace cldnn {
namespace cpu {

struct broadcast_impl : public typed_primitive_impl<broadcast> {
    using parent = typed_primitive_impl<broadcast>;
    using parent::parent;

    ov::Shape target_shape;
    ov::op::BroadcastModeSpec broadcast_mode;
    std::vector<size_t> axes_mapping;

    std::shared_ptr<ov::op::v3::Broadcast> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::broadcast_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<broadcast_impl>(*this);
    }

    broadcast_impl() : parent("broadcast_cpu_impl") {}

    explicit broadcast_impl(const broadcast_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<broadcast>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<broadcast>();
        broadcast_mode = node.get_primitive()->broadcast_mode;
        target_shape = node.get_primitive()->target_shape;
        auto axes_mapping_set = node.get_primitive()->axes_mapping;
        axes_mapping = std::vector<size_t>(axes_mapping_set.begin(), axes_mapping_set.end());
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << make_data(&broadcast_mode, sizeof(ov::op::BroadcastModeSpec));
        ob << target_shape;
        ob << axes_mapping;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> make_data(&broadcast_mode, sizeof(ov::op::BroadcastModeSpec));
        ib >> target_shape;
        ib >> axes_mapping;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, broadcast_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "broadcast::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }

        auto params = instance.get_impl_params();

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        if (!op) {
            op = std::make_shared<ov::op::v3::Broadcast>();
            op->set_broadcast_spec(broadcast_mode);

            OPENVINO_ASSERT(op->has_evaluate(), "[GPU] Couldn't find evaluate() function for broadcast ",
                                                "primitive with id ", instance.id());
        }

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        if (instance.dependencies().size() < 2) {
            OPENVINO_ASSERT(!target_shape.empty(), "[GPU] Unexpected empty target_shape for broadcast operation with id ", instance.id());
            input_host_tensors.push_back(ov::Tensor(ov::element::Type_t::i64, {target_shape.size()}, target_shape.data()));
        }

        if (instance.dependencies().size() < 3 && broadcast_mode == ov::op::BroadcastType::EXPLICIT) {
            OPENVINO_ASSERT(!axes_mapping.empty(), "[GPU] Unexpected empty axes_mapping for broadcast operation with id ", instance.id());
            input_host_tensors.push_back(ov::Tensor(ov::element::Type_t::i64, {axes_mapping.size()}, axes_mapping.data()));
        }

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);
        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute broadcast primitive with id ", instance.id());

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

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
    static std::unique_ptr<primitive_impl> create(const broadcast_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<broadcast_impl>();
    }
};


namespace detail {

attach_broadcast_impl::attach_broadcast_impl() {
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

    implementation_map<broadcast>::add(impl_types::cpu, shape_types::static_shape, broadcast_impl::create, types, formats);
    implementation_map<broadcast>::add(impl_types::cpu, shape_types::dynamic_shape, broadcast_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::broadcast_impl)
