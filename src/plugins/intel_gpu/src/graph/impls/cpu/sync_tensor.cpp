// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_tensor_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "openvino/core/parallel.hpp"
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
        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        sub_mem_mgr->set_memory_used(id, w_rank);
        while (true) {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            if (sub_mem_mgr->_use_count[id] == w_size) {
                sub_mem_mgr->_use_count[id] = 0;
                for (size_t i = 0; i < w_size; i++) {
                    sub_mem_mgr->_memorys_table[id][i].flag = false;
                }
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
        }
        stream.finish();
        sub_mem_mgr->_memorys_table[id][w_rank].send_buf = instance.input_memory().buffer_ptr();
        sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        auto dst_ptr = static_cast<uint8_t*>(instance.output_memory().buffer_ptr());
        std::vector<int> wait_list(w_size, 1);
        auto dims = instance.output_memory_ptr()->get_layout().get_dims();
        const int dim = dims.size() - 1;
        auto element_size = ov::element::Type(instance.output_memory_ptr()->get_layout().data_type).size();
        auto channel_size = dims[dim] * element_size;    // selected dim bytes
        // const int step = (mem_size / channel_size);     // the steps need to copy.
        const size_t count = (instance.output_memory_ptr()->size() / channel_size);     // the steps need to copy.
        const auto copySize = (dims[dim] / w_size) * element_size;
        while (true) {
            int wait_size = 0;
            for (size_t idx = 0; idx < w_size; idx++) {
                if (wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    auto new_ptr = static_cast<uint8_t*>(sub_mem_mgr->_memorys_table[id][idx].send_buf);
                    // parallel_for(step, [&](int i) {
                    //     int src_offset = i * copySize;
                    //     int dst_offset = i * copySize * 2 + idx * copySize;
                    //     cpu_parallel_memcpy(dst_ptr + dst_offset, new_ptr + src_offset, copySize);
                    // });
                    const size_t unloop = 8;
                    size_t step = count / unloop;
                    ov::parallel_for(step, [&](size_t i){
                        // int src_offset = i * copySize;
                        // int dst_offset = i * copySize * 2 + idx * copySize;
                        std::memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop)),     new_ptr + copySize * (i * unloop), copySize);
                        std::memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 1)), new_ptr + copySize * (i * unloop + 1), copySize);
                        std::memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 2)), new_ptr + copySize * (i * unloop + 2), copySize);
                        std::memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 3)), new_ptr + copySize * (i * unloop + 3), copySize);
                        std::memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 4)), new_ptr + copySize * (i * unloop + 4), copySize);
                        std::memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 5)), new_ptr + copySize * (i * unloop + 5), copySize);
                        std::memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 6)), new_ptr + copySize * (i * unloop + 6), copySize);
                        std::memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 7)), new_ptr + copySize * (i * unloop + 7), copySize);
                    });
                    size_t tail = count & ~(unloop - 1);
                    for (size_t i = tail; i < count; ++i) {
                        int src_offset = i * copySize;
                        int dst_offset = i * copySize * 2 + idx * copySize;
                        std::memcpy(dst_ptr + dst_offset, new_ptr + src_offset, copySize);
                    }
                    wait_list[idx] = 0;
                }
                wait_size += wait_list[idx];
            }
            if (wait_size == 0) {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_use_count[id]++;
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