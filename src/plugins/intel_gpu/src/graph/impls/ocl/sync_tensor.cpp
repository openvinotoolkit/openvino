// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "sync_tensor_inst.h"
#include "sync_tensor/sync_tensor_kernel_selector.h"
#include "sync_tensor/sync_tensor_kernel_ref.h"
#include "runtime/level_zero/ocl_context.h"
#include "runtime/level_zero/lz_context.h"


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

    void get_contexts(sync_tensor_inst& instance, oclContext& oclctx, lzContext& lzctx) {
        std::cout << "[gpu_p2p] start " << std::endl;

        auto cldnnDevice = instance.get_network().get_engine().get_device();
        auto oclDevice = std::dynamic_pointer_cast<ocl::ocl_device>(cldnnDevice);
        auto clDevice = oclDevice->get_device();
        cl_device_id device_id = clDevice();

        int device_idx = oclctx.get_device_idx(device_id);

        std::cout << "[gpu_p2p] create oclctx " << std::endl;
        oclctx.init(device_idx);
        std::cout << "[gpu_p2p] create lzContext " << std::endl;
        lzctx.initZe(device_idx);
    }

    // void* create_lz_buff(oclContext& oclctx, lzContext& lzctx, void* buff, size_t elemCount, size_t length) {
    void* create_lz_buff(oclContext& oclctx, lzContext& lzctx, std::vector<uint32_t>& initBuf, size_t elemCount) {
        std::cout << "[gpu_p2p] create_lz_buff " << std::endl;

        printf("initBuf[0]: %d \n", initBuf[0]);
        cl_mem clbuf = oclctx.createBuffer(elemCount * sizeof(uint32_t), initBuf);
        // oclctx.printBuffer(clbuf);

        // derive the dma-buf handles from opencl buffers
        uint64_t handle = oclctx.deriveHandle(clbuf);
        void *lzptr = lzctx.createFromHandle(handle, elemCount * sizeof(uint32_t));
        lzctx.printBuffer(lzptr);

        return lzptr;
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
        stream.finish();

        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        printf("[sync_tensor_impl:%d] memory id: %d\n", w_rank, id);
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

        oclContext oclctx;
        lzContext lzctx;
        size_t elemCount = 1024 * 1024;
        get_contexts(instance, oclctx, lzctx);

        std::vector<uint32_t> initBuf(elemCount, 0);
        {
            union TypeValue {
                float f;
                uint32_t i;
            };
            size_t copy_len = instance.output_memory((w_rank + 1) % 2).size();
            size_t typed_copy_len = copy_len;
            printf("[sync_tensor_impl:%d] copy_len:%ld \n", w_rank, copy_len);

            auto sec_mem = instance.output_memory_ptr(w_rank);
            auto data_layout = instance.get_output_layout(w_rank);
            std::cout << "[sync_tensor_impl:" << w_rank << "] data_layout: " << data_layout << std::endl;

            auto actual_mem = sec_mem->get_engine()->reinterpret_buffer(*sec_mem, data_layout);
            auto mem_dt = actual_mem->get_layout().data_type;
            if (mem_dt == cldnn::data_types::f16) {
                typed_copy_len /= 2;
                printf("[create_local_buff:%d] local mem_dt: cldnn::data_types::f16 \n", w_rank);
            }
            mem_lock<ov::float16, mem_lock_type::read> lock(actual_mem, stream);
            auto mem_ptr = lock.data();

            for (size_t i = 0; i < typed_copy_len; ++i) {
                TypeValue val;
                val.f = static_cast<float>(mem_ptr[i]);
                printf("[create_local_buff:%d] local val.f: %f, val.i: %u \n", w_rank, val.f, val.i);

                initBuf[i] = val.i;
                printf("local initBuf[%ld]: %u \n", i, initBuf[0]);
            }
        }

        // void *lz_buff = create_lz_buff(oclctx, lzctx, buff, elemCount, length);
        void *lz_buff = create_lz_buff(oclctx, lzctx, initBuf, elemCount);
        printf("[sync_tensor_impl:%d] lzctx.printBuffer lz_buff \n", w_rank);
        lzctx.printBuffer(lz_buff);

        sub_mem_mgr->_memorys_table[id][w_rank].send_buf = lz_buff;
        printf("[sync_tensor_impl:%d] send_buf: %p\n", w_rank, lz_buff);
        sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        std::vector<int> wait_list(w_size, 1);
        wait_list[w_rank] = 0; // no need to wait for itself
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    // auto src_ptr = static_cast<uint8_t*>(sub_mem_mgr->_memorys_table[id][idx].send_buf);
                    auto dst_ptr = instance.output_memory(idx).buffer_ptr();
                    size_t dst_len = instance.output_memory(idx).size();
                    size_t typed_dst_len = dst_len;
                    printf("[sync_tensor_impl:%d] dst_ptr:%p dst_len:%ld \n", w_rank, dst_ptr, dst_len);
                    // std::memcpy(dst_ptr, src_ptr, instance.output_memory(idx).size());

                    void* remote_send_buf = (sub_mem_mgr->_memorys_table[id][idx].send_buf);
                    printf("[sync_tensor_impl:%d] remote_send_buf: %p \n", w_rank, remote_send_buf);
                    // void *local_buff = create_lz_buff(oclctx, lzctx, nullptr, elemCount, 0);
                    std::vector<uint32_t> emptyBuf(elemCount, 0);
                    void *local_buff = create_lz_buff(oclctx, lzctx, emptyBuf, elemCount);
                    printf("[sync_tensor_impl:%d] lzctx.printBuffer local_buff \n", w_rank);
                    lzctx.printBuffer(local_buff);

                    // copy from remote level_zero buff
                    std::cout << "[-->] sync_tensor_impl w_rank: " << w_rank << ", runKernel " << std::endl;
                    lzctx.runKernel("./test_kernel_dg2.spv", "local_read_from_remote", remote_send_buf, local_buff, elemCount);

                    printf("[sync_tensor_impl:%d] after runKernel lzctx.printBuffer local_buff \n", w_rank);
                    lzctx.printBuffer(local_buff);

                    std::vector<uint32_t> outBuf(elemCount, 0);
                    lzctx.readBuffer(outBuf, local_buff, elemCount * sizeof(uint32_t));
                    printf("[sync_tensor_impl:%d] readBuffer \n", w_rank);
                    for (size_t i = 0; i < 16; ++i) {
                        printf("[%d, %f] ", outBuf[i], static_cast<float>(outBuf[i]));
                    }
                    printf("\n");

                    size_t output_idx = idx;
                    printf("[sync_tensor_impl:%d] check output %ld \n", w_rank, output_idx);
                    auto sec_mem = instance.output_memory_ptr(output_idx);
                    auto data_layout = instance.get_output_layout(output_idx);
                    printf("[sync_tensor_impl:%d] reinterpret_buffer \n", w_rank);
                    auto actual_mem = sec_mem->get_engine()->reinterpret_buffer(*sec_mem, data_layout);
                    auto mem_dt = actual_mem->get_layout().data_type;
                    if (mem_dt == cldnn::data_types::f16) {
                        typed_dst_len /= 2;
                        printf("[sync_tensor_impl:%d] cldnn::data_types::f16 \n", w_rank);
                    }
                    printf("[sync_tensor_impl:%d] lock \n", w_rank);
                    mem_lock<ov::float16, mem_lock_type::read_write> lock2(actual_mem, stream);
                    printf("[sync_tensor_impl:%d] get pointer \n", w_rank);
                    auto mem_ptr2 = lock2.data();
                    auto sec_val2 = static_cast<float>(mem_ptr2[0]);
                    printf("[sync_tensor_impl:%d] sec_val2: %f \n", w_rank, sec_val2);

                    union TypeValue {
                        float f;
                        uint32_t i;
                    };

                    for (size_t i = 0; i < typed_dst_len; ++i) {
                        TypeValue val;
                        val.i = outBuf[i];
                        printf("[sync_tensor_impl:%d] val.i: %d, val.f: %f \n", w_rank, val.i, val.f);

                        mem_ptr2[i] = static_cast<ov::float16>(val.f);
                        sec_val2 = static_cast<float>(mem_ptr2[i]);
                        printf("[sync_tensor_impl:%d] sec_val2: %f \n", w_rank, sec_val2);
                    }

                    printf("[sync_tensor_impl:%d] std::memcpy end \n", w_rank);
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
        shape_types::static_shape,
        typed_primitive_impl_ocl<sync_tensor>::create<sync_tensor_impl>, keys);
}


}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)
