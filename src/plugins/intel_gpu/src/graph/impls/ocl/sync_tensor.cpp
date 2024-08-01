// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "sync_tensor_inst.h"
#include "sync_tensor/sync_tensor_kernel_selector.h"
#include "sync_tensor/sync_tensor_kernel_ref.h"
#include "runtime/level_zero/ocl_context.h"
#include "runtime/level_zero/lz_context.h"
#include "runtime/ocl/ocl_memory.hpp"


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
        auto cldnnDevice = instance.get_network().get_engine().get_device();
        auto oclDevice = std::dynamic_pointer_cast<ocl::ocl_device>(cldnnDevice);
        auto clDevice = oclDevice->get_device();
        cl_device_id device_id = clDevice();
        printf("[get_contexts] device_id: %p \n", device_id);

        int device_idx = oclctx.get_device_idx(device_id);

        std::cout << "[gpu_p2p] create oclctx " << std::endl;
        oclctx.init(device_idx);
        std::cout << "[gpu_p2p] create lzContext " << std::endl;
        lzctx.initZe(device_idx);
    }

    void* create_lz_buff(oclContext& oclctx, lzContext& lzctx, std::vector<uint32_t>& initBuf, const size_t elemCount, const int rank) {
        cl_mem clbuf = oclctx.createBuffer(elemCount * sizeof(uint32_t), initBuf);
        printf("[create_lz_buff:%d] clbuf: %p \n", rank, clbuf);
        // oclctx.printBuffer(clbuf);

        // derive the dma-buf handles from opencl buffers
        uint64_t handle = oclctx.deriveHandle(clbuf);
        void *lzptr = lzctx.createFromHandle(handle, elemCount * sizeof(uint32_t));
        printf("[create_lz_buff:%d] lzptr: %p \n", rank, lzptr);
        lzctx.printBuffer(lzptr);

        return lzptr;
    }

    void printCLBuff(cl_command_queue& cl_queue, cl_mem& clbuf, size_t size, oclContext& oclctx, std::vector<uint32_t>& resBuf) {
        printf("[printCLBuff] clbuf: %p \n", clbuf);

        cl_int err;
        err = clEnqueueReadBuffer(cl_queue, clbuf, CL_TRUE, 0, size, resBuf.data(), 0, NULL, NULL);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");
        clFinish(cl_queue);
        printf("The first %ld elements in cl_mem = %p are: \n", size, clbuf);

        uint8_t* res_ptr = reinterpret_cast<uint8_t*>(resBuf.data());
        for (size_t i = 0; i < size; i++) {
            if (i > 16 && i < size - 16) continue;
            printf("0x%x ", res_ptr[i]);
            if (i && i % 16 == 0)
                printf("\n");
        }
        printf("\n");
    }

    void printLZBuff(lzContext& lzctx, void* ptr, size_t size) {
        printf("[printLZBuff] ptr: %p \n", ptr);

        std::vector<uint32_t> resBuf(size / 4, 0);
        lzctx.readBuffer(resBuf, ptr, size);

        uint8_t* res_ptr = reinterpret_cast<uint8_t*>(resBuf.data());
        for (size_t i = 0; i < size; i++) {
            if (i > 16 && i < size - 16) continue;

            printf("[0x%x] ", res_ptr[i]);
            if (i && i % 16 == 0)
                printf("\n");
        }
        printf("\n");
    }

    void* ocl_to_lz(sync_tensor_inst& instance, stream& stream, const int idx, oclContext& oclctx, lzContext& lzctx, const size_t size, const int rank) {
        printf("[ocl_to_lz:%d] size: %ld \n", rank, size);

        cl::CommandQueue clQueue = downcast<ocl::ocl_stream>(stream).get_cl_queue();
        cl_command_queue cl_queue = clQueue();
        cl::Buffer clBuf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(idx))->get_buffer();
        cl_mem clbuf = clBuf();

        cl_int err;
        // print clbuf
        std::vector<uint32_t> resBuf(size / 4, 0);
        printf("[ocl_to_lz:%d] print output cl_mem \n", rank);
        printCLBuff(cl_queue, clbuf, size, oclctx, resBuf);

        uint64_t nativeHandle;
        err = clGetMemObjectInfo(clbuf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(nativeHandle), &nativeHandle, NULL);
        CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_ALLOCATION_HANDLE_INTEL failed");
        // printf("[ocl_to_lz] nativeHandle: %ld, 0x%lx \n", nativeHandle, nativeHandle);
        // std::cout << "[ocl_to_lz] convert nativeHandle to lz handle " << std::endl;
        void *lzptr = lzctx.createFromHandle(nativeHandle, size);
        // printf("[ocl_to_lz] lzptr: %p \n", lzptr);
        // lzctx.printBuffer(lzptr);
        printf("[ocl_to_lz:%d] print output lzptr\n", rank);
        printLZBuff(lzctx, lzptr, size);

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
                    sub_mem_mgr->_memorys_table[id][i].flag_written = false;
                }
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
        }

        union TypeValue {
            float f;
            uint32_t i;
        };
        oclContext oclctx;
        lzContext lzctx;
        const size_t elemCount = 1024 * 1024;
        get_contexts(instance, oclctx, lzctx);

        std::vector<uint32_t> initBuf(elemCount, 0);
        int copy_idx = (w_rank + 1) % w_size;

        size_t copy_len = instance.output_memory(copy_idx).size();
        printf("[sync_tensor_impl:%d] copy_len: %ld \n", w_rank, copy_len);

        // // lz buff from initBuf
        // void *lz_buff = create_lz_buff(oclctx, lzctx, initBuf, elemCount);
        // lz buff from cl_mem
        void *lz_buff = ocl_to_lz(instance, stream, w_rank, oclctx, lzctx, copy_len, w_rank);
        printf("[sync_tensor_impl:%d] lz_buff: %p\n", w_rank, lz_buff);

        // printf("[sync_tensor_impl:%d] lzctx.printBuffer lz_buff \n", w_rank);
        // lzctx.printBuffer(lz_buff);
        // sub_mem_mgr->_memorys_table[id][w_rank].send_buf = lz_buff;
        // printf("[sync_tensor_impl:%d] send_buf: %p\n", w_rank, lz_buff);

        // std::vector<uint32_t> emptyBuf(elemCount, 0);
        // void *local_buff = create_lz_buff(oclctx, lzctx, emptyBuf, elemCount);
        // // printf("[sync_tensor_impl:%d] lzctx.printBuffer local_buff \n", w_rank);
        // // lzctx.printBuffer(local_buff);

        void *receive_buff = ocl_to_lz(instance, stream, copy_idx, oclctx, lzctx, copy_len, w_rank);
        printf("[sync_tensor_impl:%d] receive_buff: %p\n", w_rank, receive_buff);

        // void* local_send_buff = instance.output_memory(copy_idx).buffer_ptr();
        // printf("[sync_tensor_impl:%d] local_send_buff: %p\n", w_rank, local_send_buff);

        // sub_mem_mgr->_memorys_table[id][w_rank].send_buf = local_buff;
        // sub_mem_mgr->_memorys_table[id][w_rank].send_buf = local_send_buff;

        // sub_mem_mgr->_memorys_table[id][w_rank].send_buf = lz_buff;
        sub_mem_mgr->_memorys_table[id][w_rank].send_buf = receive_buff;

        sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        std::vector<int> wait_list(w_size, 1);
        wait_list[w_rank] = 0; // no need to wait for itself
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    // auto src_ptr = static_cast<uint8_t*>(sub_mem_mgr->_memorys_table[id][idx].send_buf);

                    // auto dst_ptr = instance.output_memory(idx).buffer_ptr();
                    // size_t dst_len = instance.output_memory(idx).size();
                    // size_t typed_dst_len = dst_len;
                    // printf("[sync_tensor_impl:%d] dst_ptr:%p dst_len:%ld \n", w_rank, dst_ptr, dst_len);
                    // // std::memcpy(dst_ptr, src_ptr, instance.output_memory(idx).size());

                    void* remote_send_buf = sub_mem_mgr->_memorys_table[id][idx].send_buf;
                    printf("[sync_tensor_impl:%d] remote_send_buf: %p \n", w_rank, remote_send_buf);

                    // write level_zero buff to remote
                    int srcOffsetX = 0;
                    int srcOffsetY = 0;
                    int strideX = 1;
                    int strideY = 1;
                    int groud_width = 256;
                    // read
                    // lzctx.runKernel("./test_kernel_dg2.spv", "local_read_from_remote", remote_send_buf, local_buff,
                    //     elemCount, srcOffsetX, srcOffsetY, strideX, strideY, groud_width);
                    // write
                    // lzctx.runKernel("./test_kernel_dg2.spv", "local_write_to_remote", remote_send_buf, lz_buff,
                    //     elemCount, srcOffsetX, srcOffsetY, strideX, strideY, groud_width);

                    printf("[sync_tensor_impl:%d] runKernel \n", w_rank);
                    // elemCount -> size
                    lzctx.runKernel("./test_kernel_dg2.spv", "local_write_to_remote", remote_send_buf, lz_buff,
                        copy_len, srcOffsetX, srcOffsetY, strideX, strideY, groud_width);
                    // lzctx.runKernel("./test_kernel_dg2.spv", "local_read_from_remote", remote_send_buf, receive_buff,
                    //     copy_len, srcOffsetX, srcOffsetY, strideX, strideY, groud_width);
                    //     // elemCount, srcOffsetX, srcOffsetY, strideX, strideY, groud_width);

                    // printf("[sync_tensor_impl:%d] after runKernel lzctx.printBuffer local_buff \n", w_rank);
                    // lzctx.printBuffer(local_buff);

                    // printf("[sync_tensor_impl:%d] std::memcpy end \n", w_rank);
                    wait_list[idx] = 0;
                    sub_mem_mgr->_memorys_table[id][w_rank].flag_written = true;
                    printf("[sync_tensor_impl:%d] set flag_written true \n", w_rank);
                }
                wait_size += wait_list[idx];
            }
            if (wait_size == 0) {
                break;
            }
        }

        // wait remote write finish, then write to local output
        std::vector<int> wait_list_remote(w_size, 1);
        wait_list_remote[w_rank] = 0; // no need to wait for itself
        while (true) {
            int idx = (w_rank + 1) % w_size;
            if (wait_list_remote[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag_written) {
                printf("[sync_tensor_impl:%d] remote write is done \n", w_rank);

                // // // std::vector<uint32_t> outBuf(elemCount, 0);
                // // // lzctx.readBuffer(outBuf, local_send_buff, elemCount * sizeof(uint32_t));
                // // std::vector<uint8_t> outBuf(copy_len, 0);
                // // lzctx.readBuffer(outBuf, local_send_buff, copy_len * sizeof(uint8_t));
                // // printf("[sync_tensor_impl:%d] readBuffer local_send_buff \n", w_rank);
                // // uint16_t* out_ptr = reinterpret_cast<uint16_t*>(outBuf.data());
                // // for (size_t i = 0; i < copy_len / 2; ++i) {
                // //     printf("[%d] ", [i]);
                // // }
                // // printf("\n");

                // Size_t dst_len = instance.output_memory(idx).size();
                // Auto sec_mem = instance.output_memory_ptr(idx);
                // Auto data_layout = instance.get_output_layout(idx);
                // Printf("[sync_tensor_impl:%d] reinterpret_buffer \n", w_rank);
                // Auto actual_mem = sec_mem->get_engine()->reinterpret_buffer(*sec_mem, data_layout);
                // Auto mem_dt = actual_mem->get_layout().data_type;
                // If (mem_dt == cldnn::data_types::f16) {
                //     // typed_dst_len /= 2;
                //     printf("[sync_tensor_impl:%d] cldnn::data_types::f16 \n", w_rank);
                // }
                // Printf("[sync_tensor_impl:%d] check output %d, dst_len: %ld \n", w_rank, idx, dst_len);

                // Printf("[sync_tensor_impl:%d] start to lock memory \n", w_rank);
                // // mem_lock<ov::float16, mem_lock_type::read_write> lock(actual_mem, stream);
                // Mem_lock<ov::float16, mem_lock_type::read> lock(actual_mem, stream);
                // Auto mem_ptr = lock.data();

                // For (size_t i = 0; i < dst_len; ++i) {
                //     if (i > 16 && i < dst_len - 16) continue;

                //     TypeValue val;
                //     // read
                //     val.f = static_cast<float>(mem_ptr[i]);
                //     initBuf[i] = val.i;
                //     printf("[sync_tensor_impl:%d] check after 1 val.f: %f, val.i: %u; initBuf[%ld]: %u \n", w_rank, val.f, val.i, i, initBuf[i]);

                // }

                printf("[sync_tensor_impl:%d] check kernel receive_buff, copy_len: %ld \n", w_rank, copy_len);
                printLZBuff(lzctx, receive_buff, copy_len);

                wait_list_remote[idx] = 0;
                printf("[sync_tensor_impl:%d] wirte to local done, break \n", w_rank);
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
