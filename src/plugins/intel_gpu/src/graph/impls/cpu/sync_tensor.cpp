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
        auto& engine = instance.get_network().get_engine();
        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.get_node().is_in_shape_of_subgraph();
        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }
        auto split_parts = [](int len, int n) {
            int average = len / n;
            std::vector<int> parts(n, average);
            parts.back() = len - average * (n - 1);
            return parts;
        };
        auto reshape_to_2d = [](const ov::PartialShape& shape, int64_t feature) {
            auto staticShape = shape.to_shape();
            size_t total = std::accumulate(staticShape.begin(), staticShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
            std::vector<int64_t> reshapeSize = { static_cast<int64_t>(total) / feature, feature };
            return reshapeSize;
        };
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
        auto input_ps = instance.input_memory().get_layout().get_partial_shape();
        auto output_ps = instance.output_memory().get_layout().get_partial_shape();
        auto splited_dim_vec = split_parts(output_ps.to_shape()[-1], w_size);
        auto prec_size = instance.output_memory().size() / instance.output_memory().count();
        auto dst_ptr = static_cast<uint8_t*>(instance.output_memory().buffer_ptr());
        auto mem_size = instance.output_memory().size();
        int64_t input_feature = input_ps[std::min(input_ps.size(), static_cast<size_t>(4)) - 1].get_length();
        int64_t output_feature = output_ps[std::min(output_ps.size(), static_cast<size_t>(4)) - 1].get_length();
        auto in_shape = reshape_to_2d(input_ps, input_feature);
        auto out_shape = reshape_to_2d(output_ps, output_feature);
        // handle potential fake alignment of successor FC layers
        if (out_shape[0] > in_shape[0]) {
            auto align_muliplex = out_shape[0] / in_shape[0];
            mem_size /= align_muliplex;
        }
        auto channel_size = instance.output_memory().get_layout().get_partial_shape().to_shape()[-1] * prec_size;
        const size_t count = mem_size / channel_size;
        const auto stride_size = splited_dim_vec[0] * prec_size;
        // a turn-around to copy usm device memory out, in case issues brought by un-lock
        sub_mem_mgr->_memorys_table[id][w_rank].buf =
            engine.allocate_memory(instance.input_memory().get_layout(), allocation_type::usm_host);
        sub_mem_mgr->_memorys_table[id][w_rank].buf->copy_from(stream, instance.input_memory());
        sub_mem_mgr->_memorys_table[id][w_rank].flag = true;

        std::vector<int> wait_list(w_size, 1);
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    auto src_ptr = static_cast<uint8_t*>(sub_mem_mgr->_memorys_table[id][idx].buf->buffer_ptr());
                    const auto copy_size = splited_dim_vec[idx] * prec_size;
                    const size_t unloop = 8;
                    size_t step = count / unloop;
                     ov::parallel_for(step, [&](int i){
                        std::memcpy(dst_ptr + idx * stride_size + (i * unloop) * channel_size, src_ptr + (i * unloop) * copy_size, copy_size);
                        std::memcpy(dst_ptr + idx * stride_size + (i * unloop + 1) * channel_size, src_ptr + (i * unloop + 1) * copy_size, copy_size);
                        std::memcpy(dst_ptr + idx * stride_size + (i * unloop + 2) * channel_size, src_ptr + (i * unloop + 2) * copy_size, copy_size);
                        std::memcpy(dst_ptr + idx * stride_size + (i * unloop + 3) * channel_size, src_ptr + (i * unloop + 3) * copy_size, copy_size);
                        std::memcpy(dst_ptr + idx * stride_size + (i * unloop + 4) * channel_size, src_ptr + (i * unloop + 4) * copy_size, copy_size);
                        std::memcpy(dst_ptr + idx * stride_size + (i * unloop + 5) * channel_size, src_ptr + (i * unloop + 5) * copy_size, copy_size);
                        std::memcpy(dst_ptr + idx * stride_size + (i * unloop + 6) * channel_size, src_ptr + (i * unloop + 6) * copy_size, copy_size);
                        std::memcpy(dst_ptr + idx * stride_size + (i * unloop + 7) * channel_size, src_ptr + (i * unloop + 7) * copy_size, copy_size);
                    });
                    size_t tail = count & ~(unloop - 1);
                    for (size_t i = tail; i < count; ++i) {
                        size_t dst_offset = i * channel_size + idx * stride_size;
                        size_t src_offset = i * copy_size;
                        std::memcpy(dst_ptr + dst_offset, src_ptr + src_offset, copy_size);
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