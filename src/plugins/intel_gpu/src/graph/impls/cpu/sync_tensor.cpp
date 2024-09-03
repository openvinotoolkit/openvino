// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define CL_VERSION_3_0 1
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "impls/registry/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/add.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "register.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "sync_tensor_inst.h"

namespace cldnn {
namespace cpu {

#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050
static std::map<int, std::string> oclErrorCode = {
    {0, "CL_SUCCESS"},
    {-1, "CL_DEVICE_NOT_FOUND"},
    {-2, "CL_DEVICE_NOT_AVAILABLE"},
    {-3, "CL_COMPILER_NOT_AVAILABLE"},
    {-4, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {-5, "CL_OUT_OF_RESOURCES"},
    {-6, "CL_OUT_OF_HOST_MEMORY"},
    {-7, "CL_PROFILING_INFO_NOT_AVAILABLE"},
    {-8, "CL_MEM_COPY_OVERLAP"},
    {-9, "CL_IMAGE_FORMAT_MISMATCH"},
    {-10, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {-11, "CL_BUILD_PROGRAM_FAILURE"},
    {-12, "CL_MAP_FAILURE"},
    {-13, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {-14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {-15, "CL_COMPILE_PROGRAM_FAILURE"},
    {-16, "CL_LINKER_NOT_AVAILABLE"},
    {-17, "CL_LINK_PROGRAM_FAILURE"},
    {-18, "CL_DEVICE_PARTITION_FAILED"},
    {-19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
    {-30, "CL_INVALID_VALUE"},
    {-31, "CL_INVALID_DEVICE_TYPE"},
    {-32, "CL_INVALID_PLATFORM"},
    {-33, "CL_INVALID_DEVICE"},
    {-34, "CL_INVALID_CONTEXT"},
    {-35, "CL_INVALID_QUEUE_PROPERTIES"},
    {-36, "CL_INVALID_COMMAND_QUEUE"},
    {-37, "CL_INVALID_HOST_PTR"},
    {-38, "CL_INVALID_MEM_OBJECT"},
    {-39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {-40, "CL_INVALID_IMAGE_SIZE"},
    {-41, "CL_INVALID_SAMPLER"},
    {-42, "CL_INVALID_BINARY"},
    {-43, "CL_INVALID_BUILD_OPTIONS"},
    {-44, "CL_INVALID_PROGRAM"},
    {-45, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {-46, "CL_INVALID_KERNEL_NAME"},
    {-47, "CL_INVALID_KERNEL_DEFINITION"},
    {-48, "CL_INVALID_KERNEL"},
    {-49, "CL_INVALID_ARG_INDEX"},
    {-50, "CL_INVALID_ARG_VALUE"},
    {-51, "CL_INVALID_ARG_SIZE"},
    {-52, "CL_INVALID_KERNEL_ARGS"},
    {-53, "CL_INVALID_WORK_DIMENSION"},
    {-54, "CL_INVALID_WORK_GROUP_SIZE"},
    {-55, "CL_INVALID_WORK_ITEM_SIZE"},
    {-56, "CL_INVALID_GLOBAL_OFFSET"},
    {-57, "CL_INVALID_EVENT_WAIT_LIST"},
    {-58, "CL_INVALID_EVENT"},
    {-59, "CL_INVALID_OPERATION"},
    {-60, "CL_INVALID_GL_OBJECT"},
    {-61, "CL_INVALID_BUFFER_SIZE"},
    {-62, "CL_INVALID_MIP_LEVEL"},
    {-63, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {-64, "CL_INVALID_PROPERTY"},
    {-65, "CL_INVALID_IMAGE_DESCRIPTOR"},
    {-66, "CL_INVALID_COMPILER_OPTIONS"},
    {-67, "CL_INVALID_LINKER_OPTIONS"},
    {-68, "CL_INVALID_DEVICE_PARTITION_COUNT"},
    {-69, "CL_INVALID_PIPE_SIZE"},
    {-70, "CL_INVALID_DEVICE_QUEUE"},
    {-71, "CL_INVALID_SPEC_ID"},
    {-72, "CL_MAX_SIZE_RESTRICTION_EXCEEDED"},
};
#define CHECK_OCL_ERROR(err, msg)                                                                            \
    if (err < 0) {                                                                                           \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n",                                      \
               __FUNCTION__,                                                                                 \
               __LINE__,                                                                                     \
               msg,                                                                                          \
               err,                                                                                          \
               errstr.c_str());                                                                              \
    }

#define CHECK_OCL_ERROR_RETURN(err, msg, ret)                                                                \
    if (err < 0) {                                                                                           \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n",                                      \
               __FUNCTION__,                                                                                 \
               __LINE__,                                                                                     \
               msg,                                                                                          \
               err,                                                                                          \
               errstr.c_str());                                                                              \
        return ret;                                                                                          \
    }

#define CHECK_OCL_ERROR_EXIT(err, msg)                                                                       \
    if (err < 0) {                                                                                           \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n",                                      \
               __FUNCTION__,                                                                                 \
               __LINE__,                                                                                     \
               msg,                                                                                          \
               err,                                                                                          \
               errstr.c_str());                                                                              \
        exit(1);                                                                                             \
    }
static std::mutex debug_mutex;
static const std::chrono::_V2::system_clock::time_point perf_dump_start() {
    return std::chrono::high_resolution_clock::now();
}
static std::once_flag check_flag;
static bool debug_enable = false;
static void perf_dump_done(const std::chrono::_V2::system_clock::time_point& start,
                           std::string str,
                           bool enable = false) {
    std::call_once(check_flag, [] {
        const char* env = getenv("OV_TP_P2P_DEBUG");
        if (env)
            debug_enable = true;
    });
    if (enable && debug_enable) {
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_1 = end - start;
        {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << str << " cost: " << elapsed_1.count() << " ms" << std::endl;
        }
    }
}

class ocl_p2p_helper {
public:
    ocl_p2p_helper() {}
    ~ocl_p2p_helper() {
        if (inited_ == true) {
            clReleaseKernel(kernel);
            clReleaseProgram(program);
        }
    }

    uint64_t derive_handle(cl_mem clbuf) {
        cl_int err;
        uint64_t fd;
        err = clGetMemObjectInfo(clbuf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(fd), &fd, NULL);
        CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_ALLOCATION_HANDLE_INTEL failed");
        return fd;
    }

    cl_mem map_remote_mem(cl_context context, cl_mem clbuf, size_t size) {
        cl_int err;
        const auto start = perf_dump_start();
        uint64_t fd = derive_handle(clbuf);

        // Create extMemBuffer of type cl_mem from fd.
        cl_mem_properties extMemProperties[] = {(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
                                                (cl_mem_properties)fd,
                                                0};
        cl_mem extMemBuffer = clCreateBufferWithProperties(context, extMemProperties, 0, size, NULL, &err);
        CHECK_OCL_ERROR(err, "clCreateBufferWithProperties - CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR failed");

        if (0) {
            cl_mem_object_type type;
            err = clGetMemObjectInfo(extMemBuffer, CL_MEM_TYPE, sizeof(type), &type, NULL);
            CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_TYPE failed");
            printf("size = %ld, remote_type = %d\n", size, type);
        }

        perf_dump_done(start, std::string("derive_map_remote_mem host time"));
        return extMemBuffer;
    }

    cl_mem map_remote_mem(cl_context context, uint64_t fd, size_t size) {
        cl_int err;
        const auto start = perf_dump_start();
        // Create extMemBuffer of type cl_mem from fd.
        cl_mem_properties extMemProperties[] = {(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
                                                (cl_mem_properties)fd,
                                                0};
        cl_mem extMemBuffer = clCreateBufferWithProperties(context, extMemProperties, 0, size, NULL, &err);
        CHECK_OCL_ERROR(err, "clCreateBufferWithProperties - CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR failed");

        perf_dump_done(start, std::string("map_remote_mem host time"));
        return extMemBuffer;
    }

    void destory_remote_mem(cl_mem clbuf) {
        clReleaseMemObject(clbuf);
    }

    void remote_write(cldnn::stream& stream, cl_mem src, cl_mem dst, size_t elemCount) {
        cl_int err;
        const auto start = perf_dump_start();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        // Use 4 bytes to copy in each work item
        const char write_kernel_code[] = R"(
            kernel void write_to_remote(const global int *src, global int *dst)
            {
                const int id = get_global_id(0);
                //printf("write_to_remote: dst[%d] = %d, src[%d] = %d\n", id, dst[id], id, src[id]);
                dst[id] = src[id];
            })";
        const char kernelName[] = "write_to_remote";

        cl_uint knlcount = 1;
        const char* knlstrList[] = {write_kernel_code};
        size_t knlsizeList[] = {strlen(write_kernel_code)};

        if (inited_ == false) {
            cl_context context = ocl_stream.get_engine().get_cl_context().get();
            program = clCreateProgramWithSource(context, knlcount, knlstrList, knlsizeList, &err);
            CHECK_OCL_ERROR_EXIT(err, "clCreateProgramWithSource failed");

            std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
            err = clBuildProgram(program, 0, NULL, buildopt.c_str(), NULL, NULL);
            if (err < 0) {
                size_t logsize = 0;
                auto device = ocl_stream.get_engine().get_cl_device().get();
                err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
                CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");

                std::vector<char> logbuf(logsize + 1, 0);
                err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), NULL);
                CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");
                printf("%s\n", logbuf.data());

                exit(1);
            }

            kernel = clCreateKernel(program, kernelName, &err);
            CHECK_OCL_ERROR_EXIT(err, "clCreateKernel failed");
            inited_ = true;
        }

        if (0) {
            // std::lock_guard<std::mutex> lock(debug_mutex);
            size_t data_size = 0;
            err = clGetMemObjectInfo(src, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL);
            CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_SIZE failed");
            printf("src: size = %ld, data_size = %ld\n", elemCount, data_size);

            err = clGetMemObjectInfo(dst, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL);
            CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_SIZE failed");
            printf("dst: size = %ld, data_size = %ld\n", elemCount, data_size);
        }

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg src failed");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
        CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg dst failed");

        size_t global_size[] = {elemCount / 4};
        auto queue = ocl_stream.get_cl_queue().get();
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, nullptr);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");
        clFinish(queue);  // todo: opt performance by removing clFinish

        perf_dump_done(start, std::string("ocl p2p host time for") + std::to_string(elemCount) + std::string(" bytes"));
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

    event::ptr remote_copy(cldnn::stream& stream, cl_mem src, cl_mem dst, size_t size) {
        const auto start = perf_dump_start();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        // auto usr_event = ocl_stream.create_user_event(false);
        // cl_event& ret = std::dynamic_pointer_cast<ocl::ocl_event>(usr_event)->get().get();
        cl_event ret;
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, size, 0, NULL, &ret);
        // auto event = ocl_stream.create_event(cl::Event(ret));
        // clFinish(queue);
        perf_dump_done(start, std::string("p2p copy host time for") + std::to_string(size) + std::string(" bytes"));
        // return usr_event;
        return ocl_stream.create_event(cl::Event(ret));
        // return stream.create_user_event(true);
    }

private:
    bool inited_ = false;
    cl_program program;
    cl_kernel kernel;
};

#define KERNEL_DATA_TYPE_NUM 3
typedef enum _kernel_data_type {
    e_type_fp16 = 0,
    e_type_int8 = 1,
    e_type_fp32 = 2,
} kernel_data_type;

inline kernel_data_type element_type_to_kernel_data_type(ov::element::Type_t element_type) {
    switch (element_type) {
    case ov::element::f16:
        return kernel_data_type::e_type_fp16;
    case ov::element::i8:
        return kernel_data_type::e_type_int8;
    case ov::element::f32:
        return kernel_data_type::e_type_fp32;
    default:
        printf("Error: unsupported element type for kernel adder - %s\n",
               ov::element::Type(element_type).to_string().c_str());
        break;
    }
    return kernel_data_type::e_type_int8;
}

static void dump_cl_buf(cl_command_queue queue, cl_mem clbuf, size_t count, size_t offset) {
    cl_int err;
    std::vector<float> outBuf(count, 0);
    err = clEnqueueReadBuffer(queue, clbuf, CL_TRUE, offset, count * 4, outBuf.data(), 0, NULL, NULL);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");
    clFinish(queue);

    printf("The first %ld elements in cl_mem = %p are: \n", count, clbuf);
    for (int i = 0; i < static_cast<int>(count); i++) {
        printf("%f, ", outBuf[i]);
        if (i && i % 16 == 0)
            printf("\n");
    }
    printf("\n\n");
}

class simple_ocl_add {
public:
    simple_ocl_add() {}
    ~simple_ocl_add() {
        if (inited_ == true) {
            if (kernels[kernel_data_type::e_type_fp16])
                clReleaseKernel(kernels[kernel_data_type::e_type_fp16]);
            if (kernels[kernel_data_type::e_type_int8])
                clReleaseKernel(kernels[kernel_data_type::e_type_int8]);
            clReleaseProgram(program);
        }
    }

    cl_kernel create_kernel(cldnn::stream& stream, const char* kernel_code, const char* kernelName) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);

        std::cout << "get_or_create_kernel_if_possible: create_kernel name = " << kernelName << std::endl;

        cl_uint knlcount = 1;
        const char* knlstrList[] = {kernel_code};
        size_t knlsizeList[] = {strlen(kernel_code)};

        cl_context context = ocl_stream.get_engine().get_cl_context().get();
        program = clCreateProgramWithSource(context, knlcount, knlstrList, knlsizeList, &err);
        CHECK_OCL_ERROR_EXIT(err, "clCreateProgramWithSource failed");

        std::cout << "get_or_create_kernel_if_possible: create_kernel->clCreateProgramWithSource " << kernelName << std::endl;

        std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
        err = clBuildProgram(program, 0, NULL, buildopt.c_str(), NULL, NULL);
        if (err < 0) {
            size_t logsize = 0;
            auto device = ocl_stream.get_engine().get_cl_device().get();
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
            CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");

            std::vector<char> logbuf(logsize + 1, 0);
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), NULL);
            CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");
            printf("%s\n", logbuf.data());

            exit(1);
        }
         std::cout << "get_or_create_kernel_if_possible: create_kernel->clBuildProgram " << kernelName << std::endl;
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        CHECK_OCL_ERROR_EXIT(err, "clCreateKernel failed");
        return kernel;
    }

    cl_kernel get_or_create_kernel_if_possible(cldnn::stream& stream, kernel_data_type type) {
        // std::cout << "get_or_create_kernel_if_possible: type = " << static_cast<int>(type) << std::endl;
        if (type == kernel_data_type::e_type_fp16) {
            if (kernels[type])
                return kernels[type];
            const char tensor_add_kernel_code_fp16[] = R"(
            kernel void tensor_add_func_fp16(const global half *src, global half *dst)
            {
                const int id = get_global_id(0);
                dst[id] += src[id];
                // printf("tensor_add_func_fp16: dst[%d] = %f, src[%d] = %f\n", id, (float*)dst[id], id, (float*)src[id]);
            })";
            const char kernel_name_fp16[] = "tensor_add_func_fp16";
            kernels[type] = create_kernel(stream, tensor_add_kernel_code_fp16, kernel_name_fp16);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_int8) {
            if (kernels[type])
                return kernels[type];
            const char tensor_add_kernel_code_int8[] = R"(
            kernel void tensor_add_func_int8(const global char *src, global char *dst)
            {
                const int id = get_global_id(0);
                dst[id] += src[id];
                // printf("tensor_add_func_int8: dst[%d] = %d, src[%d] = %d\n", id, dst[id], id, src[id]);
            })";
            const char kernel_name_int8[] = "tensor_add_func_int8";
            kernels[type] = create_kernel(stream, tensor_add_kernel_code_int8, kernel_name_int8);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_fp32) {
            if (kernels[type])
                return kernels[type];
            const char tensor_add_kernel_code_fp32[] = R"(
            kernel void tensor_add_func_fp32(const global float *src, global float *dst)
            {
                const int id = get_global_id(0);
                dst[id] += src[id];
                // printf("tensor_add_func_fp32: dst[%d] = %f, src[%d] = %f\n", id, dst[id], id, src[id]);
            })";
            const char kernel_name_fp32[] = "tensor_add_func_fp32";
            kernels[type] = create_kernel(stream, tensor_add_kernel_code_fp32, kernel_name_fp32);
            return kernels[type];
        } else {
            printf("error: unsuport adder kernel data type %d\n", static_cast<int>(type));
            exit(1);
        }

        return kernels[type];
    }

    event::ptr tensor_add(cldnn::stream& stream,
                          cl_mem src,
                          cl_mem dst,
                          size_t element_count,
                          kernel_data_type data_type) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        const auto start = perf_dump_start();

        cl_kernel kernel = get_or_create_kernel_if_possible(stream, data_type);

        perf_dump_done(start, std::string("get_or_create_kernel_if_possible"), false);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg src failed");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
        CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg dst failed");

        perf_dump_done(start, std::string("tensor_add->clSetKernelArg"), false);

        size_t global_size[] = {element_count};
        auto queue = ocl_stream.get_cl_queue().get();
        // auto wait_event = std::dynamic_pointer_cast<ocl::ocl_event>(event)->get().get();
        // auto usr_event = ocl_stream.create_user_event(false);
        // cl_event ret = std::dynamic_pointer_cast<ocl::ocl_event>(usr_event)->get().get();
        cl_event ret;

        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, &ret);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");
        // clFinish(queue);
        // clWaitForEvents(1, &ret);

        perf_dump_done(start, std::string("tensor add host time"), true);
        return ocl_stream.create_event(cl::Event(ret));
        // return usr_event;
        // return stream.create_user_event(true);
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

private:
    bool inited_ = false;
    cl_program program;
    cl_kernel kernels[KERNEL_DATA_TYPE_NUM];
};

static ocl_p2p_helper p2p_helper_instances[4];  // max substream number
static simple_ocl_add adder_instances[4];

inline ocl_p2p_helper& get_ocl_p2p_instance(size_t id) {
    return p2p_helper_instances[id];
}

inline simple_ocl_add& get_adder_instance(size_t id) {
    return adder_instances[id];
}

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

    void wait_p2p_done(ov::intel_gpu::SubMemoryManager::ptr& sub_mem_mgr, int id, size_t w_size, int32_t w_rank) {
        // wait for all reduce data are ready
        std::vector<int> copy_list(w_size, 1);
        copy_list[w_rank] = 0;
        auto start = perf_dump_start();
        while (true) {
            for (size_t idx = 0; idx < w_size; idx++) {
                if (idx != static_cast<size_t>(w_rank) && copy_list[idx]) {
                    // Wait for reduce data has been ready
                    // auto& remote_ocl_stream =
                    //    downcast<ocl::ocl_stream>(*sub_mem_mgr->_memorys_table[id][idx].stream_ptr);
                    auto event = sub_mem_mgr->_memorys_table[id][idx].events[w_rank];
                    if (event) {
                        // remote_ocl_stream.finish();
                        event->wait();
                        copy_list[idx] = 0;
                        {
                            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                            sub_mem_mgr->_memorys_table[id][idx].events[w_rank] = nullptr;
                        }
                    }
                }
            }
            auto left_size =
                std::accumulate(copy_list.begin(), copy_list.end(), static_cast<size_t>(1), std::multiplies<size_t>());
            if (left_size == 0)
                break;
            auto end = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end - start;
            if (duration.count() > 10000) {
                std::cout << "Warning: sync_tensor wait_p2p_done timeout..." << std::endl;
            }
        }
        perf_dump_done(start,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait p2p done"),
                       true);
    }

    std::vector<cldnn::event::ptr> propagate_p2p_events(ov::intel_gpu::SubMemoryManager::ptr& sub_mem_mgr,
                                                        int id,
                                                        size_t w_size,
                                                        int32_t w_rank) {
        std::vector<cldnn::event::ptr> sync_events;
        std::vector<int> copy_list(w_size, 1);
        copy_list[w_rank] = 0;
        auto start = perf_dump_start();
        while (true) {
            for (size_t idx = 0; idx < w_size; idx++) {
                if (idx != static_cast<size_t>(w_rank) && copy_list[idx]) {
                    auto event = sub_mem_mgr->_memorys_table[id][idx].events[w_rank];
                    if (event) {
                        sync_events.emplace_back(event);
                        copy_list[idx] = 0;
                    }
                }
            }
            auto left_size =
                std::accumulate(copy_list.begin(), copy_list.end(), static_cast<size_t>(1), std::multiplies<size_t>());
            if (left_size == 0)
                break;
            auto end = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end - start;
            if (duration.count() > 10000) {
                std::cout << "Warning: sync_tensor propagate_p2p_events timeout..." << std::endl;
            }
        }

        return sync_events;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& stream = instance.get_network().get_stream();
        const bool pass_through_events = false;
        //    (stream.get_queue_type() == QueueTypes::out_of_order) && instance.get_node().is_in_shape_of_subgraph();

        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto start = perf_dump_start();
        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }
        perf_dump_done(start,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait events"),
                       true);

        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        sub_mem_mgr->set_memory_used(id, w_rank);
        auto start_1 = perf_dump_start();
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
            auto end_1 = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end_1 - start_1;
            if (duration.count() > 10000) {
                std::cout << "Warning: sync_tensor wait data ready timeout..." << std::endl;
            }
        }
        perf_dump_done(start_1,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait data ready"),
                       true);

        auto& p2p_helper = get_ocl_p2p_instance(w_rank);
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto local_context = ocl_stream.get_engine().get_cl_context().get();
        // sub_mem_mgr->_memorys_table[id][w_rank].send_buf = instance.output_memory(w_rank).buffer_ptr();
        sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs = instance.get_output_memorys();
        sub_mem_mgr->_memorys_table[id][w_rank].flag = true;

        if (0) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            int i = 0;
            for (auto& mem : instance.get_output_memorys()) {
                auto src_cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(mem)->get_buffer().get();
                printf("Init output memory %d, rank %d\n", i++, w_rank);
                dump_cl_buf(ocl_stream.get_cl_queue().get(), src_cl_buf, mem->count(), 0);
            }
        }

        std::vector<int> wait_list(w_size, 1);
        auto start_2 = perf_dump_start();
        wait_list[w_rank] = 0;  // no need to wait for itself
        size_t data_size = 0;
        event::ptr sync_event = nullptr;
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    // auto& remote_ocl_stream =
                    // downcast<ocl::ocl_stream>(*sub_mem_mgr->_memorys_table[id][idx].stream_ptr); auto remote_context
                    // = remote_ocl_stream.get_engine().get_cl_context().get();
                    auto src_cl_buf =
                        std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(w_rank))
                            ->get_buffer()
                            .get();
                    cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][idx].recv_bufs[w_rank];
                    auto dst_cl_buf_remote =
                        std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();

                    data_size = dst_mem->size();
                    auto dst_cl_buf = p2p_helper.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                    sync_event = p2p_helper.remote_copy(stream, src_cl_buf, dst_cl_buf, data_size);
                    {
                        std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                        sub_mem_mgr->_memorys_table[id][idx].events[w_rank] = sync_event;
                    }
                    if (0) {
                        std::lock_guard<std::mutex> lock(debug_mutex);
                        printf("Write output memory (rank=%d):\n", w_rank);
                        dump_cl_buf(ocl_stream.get_cl_queue().get(), dst_cl_buf, dst_mem->count(), 0);
                    }
                    // p2p_helper.destory_remote_mem(dst_cl_buf);
                    wait_list[idx] = 0;
                }
                wait_size += wait_list[idx];
            }
            if (wait_size == 0) {
                break;
            }
            auto end_2 = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end_2 - start_2;
            if (duration.count() > 10000) {
                std::cout << "Warning: sync_tensor p2p write timeout...." << std::endl;
            }
        }

        auto str_need_add = instance.get_impl_params()->need_add ? std::string("[need_add]") : std::string("");
        perf_dump_done(start_2,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor p2p write ") +
                           std::to_string(data_size) + " bytes" + str_need_add,
                       true);

        if (0) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            int i = 0;
            for (auto& mem : instance.get_output_memorys()) {
                auto src_cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(mem)->get_buffer().get();
                printf("Output memory %d, rank %d\n", i++, w_rank);
                dump_cl_buf(ocl_stream.get_cl_queue().get(), src_cl_buf, mem->count(), 0);
            }
        }

        std::vector<cldnn::event::ptr> sync_events;
        if (instance.get_impl_params()->need_add) {
            auto start_3 = perf_dump_start();
            auto dst_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(w_rank));
            auto dst_cl_buf = dst_mem->get_buffer().get();
            // auto data_size = dst_mem->size();
            // wait for all reduce data are ready
            wait_p2p_done(sub_mem_mgr, id, w_size, w_rank);
            for (size_t idx = 0; idx < w_size; idx++) {
                if (idx != static_cast<size_t>(w_rank)) {
                    auto src_cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(idx))
                                          ->get_buffer()
                                          .get();
                    sync_event = get_adder_instance(w_rank).tensor_add(
                        stream,
                        src_cl_buf,
                        dst_cl_buf,
                        dst_mem->count(),
                        element_type_to_kernel_data_type(dst_mem->get_layout().data_type));
                    sync_events.emplace_back(sync_event);
                }
            }
            // add_worker.finish(stream);
            perf_dump_done(start_3,
                           std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor allreduce add"),
                           true);

            if (0) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                printf("Add output memory (rank=%d):\n", w_rank);
                dump_cl_buf(ocl_stream.get_cl_queue().get(), dst_cl_buf, dst_mem->count(), 0);
            }
        } else {
            sync_events = propagate_p2p_events(sub_mem_mgr, id, w_size, w_rank);
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
        perf_dump_done(start,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor total"),
                       true);

        for (auto& evt : sync_events) {
            evt->wait();
        }
        return stream.create_user_event(true);
        // return sync_events.size() > 0 ? stream.group_events(sync_events) : stream.create_user_event(true);
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
