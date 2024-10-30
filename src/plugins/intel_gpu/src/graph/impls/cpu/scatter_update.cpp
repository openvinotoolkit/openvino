// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "scatter_update_inst.h"
#include "impls/registry/implementation_map.hpp"

#include "openvino/op/scatter_update.hpp"

namespace cldnn {
namespace cpu {

struct scatter_update_impl : public typed_primitive_impl<scatter_update> {
    using parent = typed_primitive_impl<scatter_update>;
    using parent::parent;

    int64_t axis = 0;

    std::shared_ptr<ov::op::v3::ScatterUpdate> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::scatter_update_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scatter_update_impl>(*this);
    }

    scatter_update_impl() : parent("scatter_update_cpu_impl") {}

    explicit scatter_update_impl(const scatter_update_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<scatter_update>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<scatter_update>();
        axis = node.get_primitive()->axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << axis;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> axis;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, scatter_update_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "scatter_update::execute_impl");
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

        auto axis_tensor = ov::Tensor(ov::element::i64, ov::Shape{1}, static_cast<void*>(&axis));

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        input_host_tensors.push_back(axis_tensor);

        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        if (!op) {
            op = std::make_shared<ov::op::v3::ScatterUpdate>();
        }

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute scatter_update primitive with id ", instance.id());

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
    static std::unique_ptr<primitive_impl> create(const scatter_update_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<scatter_update_impl>();
    }
};


namespace detail {

attach_scatter_update_impl::attach_scatter_update_impl() {
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

    implementation_map<scatter_update>::add(impl_types::cpu, shape_types::static_shape, scatter_update_impl::create, types, formats);
    implementation_map<scatter_update>::add(impl_types::cpu, shape_types::dynamic_shape, scatter_update_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::scatter_update_impl)
