// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_tensor_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"

namespace cldnn {
namespace cpu {

struct sync_tensor_impl : public typed_primitive_impl<sync_tensor> {
    using parent = typed_primitive_impl<sync_tensor>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::sync_tensor_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<sync_tensor_impl>(*this);
    }

    sync_tensor_impl() : parent() {}

    explicit sync_tensor_impl(const sync_tensor_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<sync_tensor>());
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.get_node().is_in_shape_of_subgraph();

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }
        memory::ptr input_mem;
        input_mem = instance.input_memory_ptr();
        auto message = ov::threading::message_manager();
        auto vec_message = message->wait_message(/*cur_rank*/w_rank, /*streams_num*/w_size);
        auto recv_buf = instance.output_memory_ptr()->buffer_ptr();
        for (int idx=0; idx < w_size; ++idx) {
            // if (idx == w_rank) {
            //     continue;
            // }
            float* send_ptr = static_cast<float*>(vec_message[idx].buf);
            //ov::Extensions::Cpu::XARCH::allreduce_float32(send_ptr, recv_buf, count);
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

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const sync_tensor_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<sync_tensor_impl>();
    }
};


namespace detail {

attach_sync_tensor_impl::attach_sync_tensor_impl() {
    implementation_map<sync_tensor>::add(impl_types::cpu, shape_types::dynamic_shape, sync_tensor_impl::create, {});
    implementation_map<sync_tensor>::add(impl_types::cpu, shape_types::static_shape, sync_tensor_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)