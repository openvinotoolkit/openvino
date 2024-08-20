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
#include <unistd.h>

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

    void get_contexts(sync_tensor_inst& instance, oclContext& oclctx, lzContext& lzctx, int rank) {
        auto start = std::chrono::high_resolution_clock::now();

        int device_idx = oclctx.device_idx();
        printf("[get_contexts] device_idx: %d \n", device_idx);
        if (device_idx == -1) {
            auto cldnnDevice = instance.get_network().get_engine().get_device();
            auto oclDevice = std::dynamic_pointer_cast<ocl::ocl_device>(cldnnDevice);
            auto clDevice = oclDevice->get_device();
            cl_device_id device_id = clDevice();
            printf("[get_contexts] device_id: %p \n", device_id);

            device_idx = oclctx.get_device_idx(device_id);
            std::cout << "[get_contexts] init oclctx " << std::endl;
            oclctx.init(device_idx);

            printf("[get_contexts] device_idx: %d \n", device_idx);
            std::cout << "[get_contexts] init lzContext " << std::endl;
            lzctx.initZe(device_idx);
            auto end_inits = std::chrono::high_resolution_clock::now();

            char cwd[256];
            getcwd(cwd, 256);
            std::string file_path = std::string(cwd) + "/test_kernel_dg2.spv";
            printf("[get_contexts] file_path: %s \n", file_path.c_str());
            lzContext::readKernel(file_path.data(), "local_write_to_remote");
            auto end_read_kernel = std::chrono::high_resolution_clock::now();

            lzctx.initKernel();
            auto end_init_kernel = std::chrono::high_resolution_clock::now();

            int64_t ts_ctx_inits = std::chrono::duration_cast<std::chrono::microseconds>(end_inits - start).count();
            int64_t ts_ctx_read_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_read_kernel - end_inits).count();
            int64_t ts_ctx_init_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_init_kernel - end_inits).count();
            printf("[sync_tensor_impl:%d] ts_ctx_inits: %ld us\n", rank, ts_ctx_inits);
            printf("[sync_tensor_impl:%d] ts_ctx_read_kernel: %ld us\n", rank, ts_ctx_read_kernel);
            printf("[sync_tensor_impl:%d] ts_ctx_init_kernel: %ld us\n", rank, ts_ctx_init_kernel);
        }
    }

    void printCLBuff(cl_command_queue& cl_queue, cl_mem& clbuf, size_t size, oclContext& oclctx, std::vector<uint32_t>& resBuf) {
        printf("[printCLBuff] clbuf: %p \n", clbuf);

        cl_int err;
        size_t main_buffer_size = 0;
        err = clGetMemObjectInfo(clbuf, CL_MEM_SIZE, sizeof(main_buffer_size), &main_buffer_size, NULL);
        CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_SIZE failed");
        printf("[printCLBuff] main_buffer_size: %ld \n", main_buffer_size);

        err = clEnqueueReadBuffer(cl_queue, clbuf, CL_TRUE, 0, size, resBuf.data(), 0, NULL, NULL);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");
        clFinish(cl_queue);
        printf("The first %ld elements in cl_mem = %p are: \n", size, clbuf);

        uint8_t* res_ptr = reinterpret_cast<uint8_t*>(resBuf.data());
        for (size_t i = 0; i < size; i++) {
            if (i > 16 && i < size - 16) continue;
            printf("0x%x ", res_ptr[i]);
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
        }
        printf("\n");
    }

    void* ocl_to_lz(sync_tensor_inst& instance, stream& stream, const int idx, oclContext& oclctx, lzContext& lzctx, const size_t size, const int rank) {
        printf("[ocl_to_lz:%d] size: %ld \n", rank, size);

        cl::CommandQueue clQueue = downcast<ocl::ocl_stream>(stream).get_cl_queue();
        // cl_command_queue cl_queue = clQueue();
        cl::Buffer clBuf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(idx))->get_buffer();
        cl_mem clbuf = clBuf();
        printf("[ocl_to_lz:%d] clbuf: %p \n", rank, clbuf);

        cl_int err;
        // print clbuf
        // std::vector<uint32_t> resBuf(size / 4, 0);
        // printf("[ocl_to_lz:%d] print output cl_mem \n", rank);
        // printCLBuff(cl_queue, clbuf, size, oclctx, resBuf);

        uint64_t nativeHandle;
        err = clGetMemObjectInfo(clbuf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(nativeHandle), &nativeHandle, NULL);
        CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_ALLOCATION_HANDLE_INTEL failed");
        // printf("[ocl_to_lz] nativeHandle: %ld, 0x%lx \n", nativeHandle, nativeHandle);
        void* lzptr = lzctx.createFromHandle(nativeHandle, size);

        // printf("[ocl_to_lz:%d] print output lzptr\n", rank);
        // printLZBuff(lzctx, lzptr, size);

        return lzptr;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        // static std::vector<std::string, double> statistics;
        auto end_post_p2p = std::chrono::high_resolution_clock::now();
        auto end_p2p_kernel = std::chrono::high_resolution_clock::now();
        auto end_pre_p2p = std::chrono::high_resolution_clock::now();
        auto end_sendbuf = std::chrono::high_resolution_clock::now();
        auto end_sync_wait = std::chrono::high_resolution_clock::now();
        auto end_contexts = std::chrono::high_resolution_clock::now();
        auto start_impl = std::chrono::high_resolution_clock::now();
        std::map<std::string, std::vector<int64_t>>& sync_tensor_timestamps = instance.get_host_timestamps();


        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.get_node().is_in_shape_of_subgraph();
        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }
        stream.finish();
        auto end_event_wait = std::chrono::high_resolution_clock::now();

        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        sub_mem_mgr->set_memory_used(id, w_rank);

        oclContext& oclctx = oclContext::getInstance(w_rank);
        lzContext& lzctx = lzContext::getInstance(w_rank);
        // const size_t elemCount = 8 * 1024; // level zero sendbuf limitation: 8KB
        get_contexts(instance, oclctx, lzctx, w_rank);
        end_contexts = std::chrono::high_resolution_clock::now();

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
        end_sync_wait = std::chrono::high_resolution_clock::now();

        size_t input_len = instance.input_memory(0).size();
        printf("[sync_tensor_impl:%d] input_len: %ld \n", w_rank, input_len);
        // std::vector<uint32_t> initBuf(elemCount, 0);
        int copy_rank = (w_rank + 1) % w_size;
        size_t copy_len = instance.output_memory(copy_rank).size();
        printf("[sync_tensor_impl:%d] copy_len: %ld \n", w_rank, copy_len);

        // std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
        // sub_mem_mgr->_memorys_table[id][w_rank].flag = false;

        if (sub_mem_mgr->_memorys_table[id][w_rank].MAX_COPY_LEN < copy_len) {
            printf("[sync_tensor_impl:%d] MAX_COPY_LEN1: %ld \n", w_rank, sub_mem_mgr->_memorys_table[id][w_rank].MAX_COPY_LEN);

            void *send_buf = ocl_to_lz(instance, stream, w_rank, oclctx, lzctx, copy_len, w_rank);
            printf("[sync_tensor_impl:%d] send_buf: %p\n", w_rank, send_buf);
            void *receive_buf = ocl_to_lz(instance, stream, copy_rank, oclctx, lzctx, copy_len, w_rank);
            printf("[sync_tensor_impl:%d] receive_buf: %p\n", w_rank, receive_buf);

            sub_mem_mgr->_memorys_table[id][w_rank].MAX_COPY_LEN = copy_len;
            printf("[sync_tensor_impl:%d] MAX_COPY_LEN2: %ld \n", w_rank, sub_mem_mgr->_memorys_table[id][w_rank].MAX_COPY_LEN);

            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_memorys_table[id][w_rank].send_buf = send_buf;
            sub_mem_mgr->_memorys_table[id][w_rank].receive_buf = receive_buf;
        }
        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        }
        end_sendbuf = std::chrono::high_resolution_clock::now();

        std::vector<int> wait_list(w_size, 1);
        wait_list[w_rank] = 0; // no need to wait for itself
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    // auto src_ptr = static_cast<uint8_t*>(sub_mem_mgr->_memorys_table[id][idx].send_buf);
                    // auto dst_ptr = instance.output_memory(idx).buffer_ptr();
                    // std::memcpy(dst_ptr, src_ptr, instance.output_memory(idx).size());

                    void* local_send_buf = sub_mem_mgr->_memorys_table[id][w_rank].send_buf;
                    void* remote_receive_buf = sub_mem_mgr->_memorys_table[id][idx].receive_buf;
                    printf("[sync_tensor_impl:%d] remote_receive_buf: %p \n", w_rank, remote_receive_buf);

                    end_pre_p2p = std::chrono::high_resolution_clock::now();

                    // write level_zero buff to remote
                    int srcOffsetX = 0;
                    int srcOffsetY = 0;
                    int strideX = 1;
                    int strideY = 1;
                    int groud_width = 256;
                    // elemCount -> size
                    printf("[sync_tensor_impl:%d] runKernel \n", w_rank);
                    lzctx.runKernel("./test_kernel_dg2.spv", "local_write_to_remote", remote_receive_buf, local_send_buf,
                        copy_len, srcOffsetX, srcOffsetY, strideX, strideY, groud_width);
                    // printf("[sync_tensor_impl:%d] print local_send_buf \n", w_rank);
                    // lzctx.printBuffer(local_send_buf);

                    end_p2p_kernel = std::chrono::high_resolution_clock::now();

                    std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                    sub_mem_mgr->_memorys_table[id][w_rank].flag_written = true;
                    printf("[sync_tensor_impl:%d] set flag_written true \n", w_rank);

                    wait_list[idx] = 0;
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
            // int idx = (w_rank + 1) % w_size;
            if (wait_list_remote[copy_rank] > 0 && sub_mem_mgr->_memorys_table[id][copy_rank].flag_written) {
                printf("[sync_tensor_impl:%d] remote write is done \n", w_rank);

                // std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                // sub_mem_mgr->_memorys_table[id][copy_rank].flag_written = false;
                // sub_mem_mgr->_memorys_table[id][copy_rank].flag = false;
                printf("[sync_tensor_impl:%d] check kernel receive_buf, copy_len: %ld \n", w_rank, copy_len);
                // printLZBuff(lzctx, receive_buf, copy_len);

                wait_list_remote[copy_rank] = 0;
                // printf("[sync_tensor_impl:%d] wirte to local done, break \n", w_rank);
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

        end_post_p2p = std::chrono::high_resolution_clock::now();

        int64_t ts_event_wait = std::chrono::duration_cast<std::chrono::microseconds>(end_event_wait - start_impl).count();
        int64_t ts_ctxs = std::chrono::duration_cast<std::chrono::microseconds>(end_contexts - end_event_wait).count();
        int64_t ts_sync_wait = std::chrono::duration_cast<std::chrono::microseconds>(end_sync_wait - end_contexts).count();
        int64_t ts_sendbuf = std::chrono::duration_cast<std::chrono::microseconds>(end_sendbuf - end_sync_wait).count();
        int64_t ts_pre_p2p = std::chrono::duration_cast<std::chrono::microseconds>(end_pre_p2p - end_sendbuf).count();
        int64_t ts_p2p_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_p2p_kernel - end_pre_p2p).count();
        int64_t ts_post_p2p = std::chrono::duration_cast<std::chrono::microseconds>(end_post_p2p - end_p2p_kernel).count();
        int64_t ts_exec = std::chrono::duration_cast<std::chrono::microseconds>(end_post_p2p - start_impl).count();
        sync_tensor_timestamps["ts_exec"].push_back(ts_exec);
        sync_tensor_timestamps["ts_event_wait"].push_back(ts_event_wait);
        sync_tensor_timestamps["ts_ctxs"].push_back(ts_ctxs);
        sync_tensor_timestamps["ts_sync_wait"].push_back(ts_sync_wait);
        sync_tensor_timestamps["ts_sendbuf"].push_back(ts_sendbuf);
        sync_tensor_timestamps["ts_pre_p2p"].push_back(ts_pre_p2p);
        sync_tensor_timestamps["ts_p2p_kernel"].push_back(ts_p2p_kernel);
        sync_tensor_timestamps["ts_post_p2p"].push_back(ts_post_p2p);
        // // printf("[sync_tensor_impl:%d] sync_tensor_timestamps[ts_exec] size: %ld \n", w_rank, sync_tensor_timestamps["ts_exec"].size());
        // // double sum = 0;
        // // double sum_avg = 0;
        // // for (size_t i = 0; i < sync_tensor_timestamps["ts_exec"].size(); ++i) {
        // //     sum += static_cast<double>(sync_tensor_timestamps["ts_exec"][i]);
        // //     sum_avg = sum / (i + 1);
        // //     printf("[sync_tensor] get_host_timestamps iter [%ld] sum: %f, sum_avg: %f \n", i, sum, sum_avg);
        // // }

        // // GPU_DEBUG_IF(debug_configuration::get_instance()->host_time_profiling) {
        printf("[sync_tensor_impl:%d] ts_sync_wait: %ld us\n", w_rank, ts_sync_wait);
        printf("[sync_tensor_impl:%d] ts_ctxs: %ld us\n", w_rank, ts_ctxs);
        // printf("[sync_tensor_impl:%d] ts_sendbuf: %ld us\n", w_rank, ts_sendbuf);
        // printf("[sync_tensor_impl:%d] ts_pre_p2p: %ld us\n", w_rank, ts_pre_p2p);
        printf("[sync_tensor_impl:%d] ts_p2p_kernel: %ld us\n", w_rank, ts_p2p_kernel);
        // printf("[sync_tensor_impl:%d] ts_post_p2p: %ld us\n", w_rank, ts_post_p2p);
        // printf("[sync_tensor_impl:%d] ts_exec: %ld us\n", w_rank, ts_exec);
        return stream.create_user_event(true);
    }

    // void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<sync_tensor>();
        auto params = get_default_params<kernel_selector::sync_tensor_params>(impl_param, is_shape_agnostic);

        // size_t rank = impl_param.get_output_layout().get_rank();
        // params.dim = get_sync_tensor_dim(primitive->dimension, rank);

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // std::cout << "[sync_tensor_impl] update_dispatch_data " << std::endl;
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
