// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "gather_inst.h"
#include "impls/registry/implementation_map.hpp"

#include "openvino/op/gather.hpp"

namespace cldnn {
namespace cpu {

struct gather_impl : public typed_primitive_impl<gather> {
    using parent = typed_primitive_impl<gather>;
    using parent::parent;

    int64_t axis = 0;
    int64_t batch_dims = 0;

    std::shared_ptr<ov::op::v8::Gather> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::gather_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_impl>(*this);
    }

    gather_impl() : parent("gather_cpu_impl") {}

    explicit gather_impl(const gather_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<gather>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<gather>();
        axis = node.get_primitive()->axis;
        batch_dims = node.get_primitive()->batch_dim;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << axis;
        ob << batch_dims;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> axis;
        ib >> batch_dims;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, gather_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "gather::execute_impl");
        auto& stream = instance.get_network().get_stream();

        if (instance.can_be_optimized()) {
            return stream.group_events(events);
        }

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
            op = std::make_shared<ov::op::v8::Gather>();
            op->set_batch_dims(batch_dims);
        }

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        auto axis_tensor = ov::Tensor(ov::element::i64, ov::Shape{1}, static_cast<void*>(&axis));

        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));
        input_host_tensors.push_back(axis_tensor);

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute gather primitive with id ", instance.id());

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
    static std::unique_ptr<primitive_impl> create(const gather_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<gather_impl>();
    }
};


namespace detail {

attach_gather_impl::attach_gather_impl() {
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

    implementation_map<gather>::add(impl_types::cpu, shape_types::static_shape, gather_impl::create, types, formats);
    implementation_map<gather>::add(impl_types::cpu, shape_types::dynamic_shape, gather_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::gather_impl)
