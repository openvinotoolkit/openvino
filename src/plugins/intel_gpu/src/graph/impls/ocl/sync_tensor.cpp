// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "sync_tensor_inst.h"
#include "sync_tensor/sync_tensor_kernel_selector.h"
#include "sync_tensor/sync_tensor_kernel_ref.h"


namespace cldnn {
namespace ocl {
static inline kernel_selector::sync_tensor_dim get_sync_tensor_dim(int64_t axis, size_t rank) {
    if (axis < 0) {
        axis += rank;
    }
    switch (axis) {
        case 0: return kernel_selector::sync_tensor_dim::BATCH;
        case 1: return kernel_selector::sync_tensor_dim::FEATURE;
        case 2:
            if (rank > 4)
                return kernel_selector::sync_tensor_dim::Z;
            else
                return kernel_selector::sync_tensor_dim::Y;
        case 3:
            if (rank > 4)
                return kernel_selector::sync_tensor_dim::Y;
            else
                return kernel_selector::sync_tensor_dim::X;
        case 4: return kernel_selector::sync_tensor_dim::X;
        default: OPENVINO_THROW("Invalid sync_tensor axis ", axis);
    }
}

struct sync_tensor_impl : public typed_primitive_impl_ocl<sync_tensor> {
    using parent = typed_primitive_impl_ocl<sync_tensor>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::sync_tensor_kernel_selector;
    using kernel_params_t = kernel_selector::sync_tensor_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::sync_tensor_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<sync_tensor_impl>(*this);
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
        std::cout << "[sync_tensor_impl] ocl execute_impl" << std::endl;

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.get_node().is_in_shape_of_subgraph();
        std::cout << "[sync_tensor_impl] ocl execute_impl: pass_through_events? " << pass_through_events << std::endl;

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }
        stream.finish();
        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        std::cout << "[-->] sync_tensor_impl execute_impl w_rank: " << w_rank << std::endl;
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        std::cout << "[-->] sync_tensor_impl execute_impl w_size: " << w_size << std::endl;
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        std::cout << "[-->] sync_tensor_impl execute_impl w_rank: " << w_rank << ", id: " << id << std::endl;
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
        sub_mem_mgr->_memorys_table[id][w_rank].send_buf = instance.output_memory(w_rank).buffer_ptr();
        std::cout << "[-->] sync_tensor_impl get send_buf w_rank: " << w_rank << std::endl;
        sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        std::cout << "[-->] sync_tensor_impl w_rank: " << w_rank << ", set flag true " << std::endl;
        std::vector<int> wait_list(w_size, 1);
        wait_list[w_rank] = 0; // no need to wait for itself
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    // auto src_ptr = static_cast<uint8_t*>(sub_mem_mgr->_memorys_table[id][idx].send_buf);
                    // auto dst_ptr = instance.output_memory(idx).buffer_ptr();
                    std::cout << "[-->] sync_tensor_impl w_rank: " << w_rank << ", memcpy, skipped" << std::endl;
                    // std::memcpy(dst_ptr, src_ptr, instance.output_memory(idx).size());
                    wait_list[idx] = 0;
                }
                wait_size += wait_list[idx];
                std::cout << "[-->] sync_tensor_impl execute_impl w_rank: " << w_rank << ", idx: " << idx << ", wait_size: " << wait_size << std::endl;
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

    // void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        std::cout << "[sync_tensor_impl] get_kernel_params is_shape_agnostic: " << is_shape_agnostic << std::endl;
        const auto& primitive = impl_param.typed_desc<sync_tensor>();
        std::cout << "[sync_tensor_impl] get_kernel_params primitive id: " << primitive->id << std::endl;
        auto params = get_default_params<kernel_selector::sync_tensor_params>(impl_param, is_shape_agnostic);

        size_t rank = impl_param.get_output_layout().get_rank();
        std::cout << "[sync_tensor_impl] get_kernel_params rank: " << rank << std::endl;
        // params.dim = get_sync_tensor_dim(primitive->dimension, rank);

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        std::cout << "[sync_tensor_impl] update_dispatch_data " << std::endl;
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }
        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};


namespace detail {

attach_sync_tensor_impl::attach_sync_tensor_impl() {
    std::cout << "[sync_tensor_impl] attach_sync_tensor_impl" << std::endl;
    auto types = {
        data_types::f32,
        data_types::f16,
    };

    auto dyn_formats = {
        format::bfyx,
    };

    auto static_formats = {
        format::bfyx,
    };

    auto keys = implementation_map<activation>::combine(types, static_formats);

    implementation_map<sync_tensor>::add(impl_types::ocl,
        shape_types::dynamic_shape,
        typed_primitive_impl_ocl<sync_tensor>::create<sync_tensor_impl>, types, dyn_formats);

    implementation_map<sync_tensor>::add(impl_types::ocl,
        // shape_types::any,
        shape_types::static_shape,
        typed_primitive_impl_ocl<sync_tensor>::create<sync_tensor_impl>, keys);
}


}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)
