// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define CL_VERSION_3_0 1
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <algorithm>
#include <mutex>
#include <condition_variable>

#include "impls/registry/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "register.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "sync_tensor_inst.h"

namespace cldnn {
namespace ocl {

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

static bool debug_enable = false;
static std::mutex debug_mutex;
static const std::chrono::_V2::system_clock::time_point perf_dump_start() {
    return std::chrono::high_resolution_clock::now();
}

static void perf_dump_done(const std::chrono::_V2::system_clock::time_point& start,
                           std::string str,
                           bool enable = false) {
    static std::once_flag check_flag;
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

class gpu_semaphore {
public:
    gpu_semaphore(int count = 2) : count_(count), total_(count) {}
    void signal() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (count_ < total_)
            ++count_;
        cv_.notify_one();
    }
    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            return count_ > 0;
        });
        --count_;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
    int total_;
};

static gpu_semaphore gpu_lock;
class gpu_p2p_helper {
public:
    gpu_p2p_helper() {}
    ~gpu_p2p_helper() {}

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
        if (err < 0) {
            OPENVINO_ASSERT(false,
                            "clCreateBufferWithProperties failed, clbuf = %p, fd = %ld, size = %ld, new_cl_mem = %p\n",
                            clbuf,
                            fd,
                            size,
                            extMemBuffer);
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

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

    void remote_copy(cldnn::stream& stream, cl_mem src, cl_mem dst, size_t size) {
        const auto start = perf_dump_start();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, size, 0, NULL, &ret);
        clWaitForEvents(1, &ret);  // blocked copy
        clReleaseEvent(ret);
        perf_dump_done(start,
                       std::string("p2p copy host time for ") + std::to_string(size) + std::string(" bytes"),
                       true);
        return;
    }
};

static void dump_cl_buf(cl_command_queue queue, cl_mem clbuf, size_t count, size_t offset) {
    cl_int err;
    std::vector<float> outBuf(count, 0);
    err = clEnqueueReadBuffer(queue, clbuf, CL_TRUE, offset, count * 4, outBuf.data(), 0, NULL, NULL);
    CHECK_OCL_ERROR(err, "clEnqueueReadBuffer failed");
    clFinish(queue);

    std::cout << "The first " << count << "elements in cl_mem = " << clbuf << " are: " << std::endl;
    for (int i = 0; i < static_cast<int>(count); i++) {
        printf("%f, ", outBuf[i]);
        std::cout << outBuf[i] << ", ";
        if (i && i % 16 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

class simple_tensor_add {
public:
    simple_tensor_add() {}
    ~simple_tensor_add() {
        for (auto& item : kernels) {
            if (item.second)
                clReleaseKernel(item.second);
        }
        kernels.clear();
        if (program)
            clReleaseProgram(program);
    }

    typedef enum _kernel_data_type {
        e_type_fp16 = 0,
        e_type_int8 = 1,
        e_type_fp32 = 2,
    } kernel_data_type;

    kernel_data_type element_type_to_kernel_data_type(ov::element::Type_t element_type) {
        switch (element_type) {
        case ov::element::f16:
            return kernel_data_type::e_type_fp16;
        case ov::element::i8:
            return kernel_data_type::e_type_int8;
        case ov::element::f32:
            return kernel_data_type::e_type_fp32;
        default:
            OPENVINO_THROW("Error: unsupported element type for kernel adder - ",
                           ov::element::Type(element_type).to_string().c_str());
            break;
        }
        return kernel_data_type::e_type_int8;
    }

    cl_kernel create_kernel(cldnn::stream& stream, const char* kernel_code, const char* kernelName) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        std::cout << "create_kernel: name = " << kernelName << std::endl;

        cl_uint knlcount = 1;
        const char* knlstrList[] = {kernel_code};
        size_t knlsizeList[] = {strlen(kernel_code)};

        cl_context context = ocl_stream.get_engine().get_cl_context().get();
        program = clCreateProgramWithSource(context, knlcount, knlstrList, knlsizeList, &err);
        CHECK_OCL_ERROR(err, "clCreateProgramWithSource failed");

        std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
        err = clBuildProgram(program, 0, NULL, buildopt.c_str(), NULL, NULL);
        if (err < 0) {
            size_t logsize = 0;
            auto device = ocl_stream.get_engine().get_cl_device().get();
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
            CHECK_OCL_ERROR(err, "clGetProgramBuildInfo failed");

            std::vector<char> logbuf(logsize + 1, 0);
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), NULL);
            std::cout << "clGetProgramBuildInfo failed: " << logbuf.data() << std::endl;
            // OPENVINO_ASSERT(err >= 0, "clGetProgramBuildInfo: ", logbuf.data());
        }
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        CHECK_OCL_ERROR(err, "clCreateKernel failed");
        return kernel;
    }

    cl_kernel get_or_create_kernel_if_possible(cldnn::stream& stream, kernel_data_type type) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = kernels.find(type);
        if (it != kernels.end()) {
            // std::cout << "get_kernel: type = " << static_cast<int>(type) << std::endl;
            return it->second;
        }
        #define ADD_OP_KERNEL_SOURCE_CODE(DATA_TYPE)                                                                       \
            "kernel void tensor_add_kernel_" #DATA_TYPE "(const global " #DATA_TYPE " *src, global " #DATA_TYPE " *dst) {" \
            "const int id = get_global_id(0);"                                                                             \
            "dst[id] += src[id];"                                                                                          \
            "}"
        if (type == kernel_data_type::e_type_fp16) {
            const char tensor_add_kernel_fp16[] = ADD_OP_KERNEL_SOURCE_CODE(half);
            const char kernel_name[] = "tensor_add_kernel_half";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp16, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_int8) {
            const char tensor_add_kernel_int8[] = ADD_OP_KERNEL_SOURCE_CODE(char);
            const char kernel_name[] = "tensor_add_kernel_char";
            kernels[type] = create_kernel(stream, tensor_add_kernel_int8, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_fp32) {
            const char tensor_add_kernel_fp32[] = ADD_OP_KERNEL_SOURCE_CODE(float);
            const char kernel_name[] = "tensor_add_kernel_float";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp32, kernel_name);
            return kernels[type];
        } else {
            std::cout << "error: unsupported adder kernel data type " << static_cast<int>(type) << std::endl;
            // OPENVINO_THROW("error: unsupported adder kernel data type ", static_cast<int>(type));
        }
        #undef ADD_OP_KERNEL_SOURCE_CODE
        return kernels[type];
    }

    event::ptr tensor_add(cldnn::stream& stream,
                          cl_mem src,
                          cl_mem dst,
                          size_t element_count,
                          kernel_data_type data_type) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        if (src == nullptr || dst == nullptr) {
            std::cout << "tensor_add: invalid arguments!" << std::endl;
        }
        OPENVINO_ASSERT(src != nullptr && dst != nullptr, "tensor_add: invalid arguments!");

        const auto start = perf_dump_start();
        cl_kernel kernel = get_or_create_kernel_if_possible(stream, data_type);
        perf_dump_done(start, std::string("get_or_create_kernel_if_possible"), false);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        CHECK_OCL_ERROR(err, "clSetKernelArg src failed");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
        CHECK_OCL_ERROR(err, "clSetKernelArg dst failed");

        size_t global_size[] = {element_count};
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, &ret);
        CHECK_OCL_ERROR(err, "clEnqueueNDRangeKernel failed");
        // clWaitForEvents(1, &ret);

        perf_dump_done(start, std::string("tensor add host time"), false);
        return ocl_stream.create_event(cl::Event(ret));
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

private:
    cl_program program;
    std::mutex mutex;
    std::map<kernel_data_type, cl_kernel> kernels;
};

class tensor_concat_memory {
public:
    tensor_concat_memory() : buf(nullptr), width(0), height(0), type(ov::element::f16) {}
    tensor_concat_memory(cl_mem _buf, size_t _w, size_t _h, size_t _stride, ov::element::Type _type)
        : buf(_buf),
          width(_w),
          height(_h),
          stride(_stride),
          type(_type) {}
    tensor_concat_memory(tensor_concat_memory& other) {
        buf = other.buf;
        width = other.width;
        height = other.height;
        stride = other.stride;
        type = other.type;
    }
    bool operator==(const tensor_concat_memory& other) const {
        return width == other.height && height == other.height && stride == other.stride;
    }

    void print() const {
        size_t data_size = 0;
        auto err = clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL);
        CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_SIZE failed");
        std::cout << "width = " << width << ", height = " << height << ", stride = " << stride
                  << ", type = " << type.to_string() << " -- actual_size = " << data_size << std::endl;
    }

    cl_mem buf;
    size_t width;
    size_t height;
    size_t stride;
    ov::element::Type type;
};

class simple_tensor_concat {
public:
    simple_tensor_concat() {}
    ~simple_tensor_concat() {}

    bool validate(std::vector<std::shared_ptr<tensor_concat_memory>>& src, std::shared_ptr<tensor_concat_memory>& dst) {
        size_t total_height = 0, total_width = 0;
        for (auto& s : src) {
            total_height += s->height;
            total_width += s->width;
            if (!s->buf)
                return false;
            if (s->type != dst->type)
                return false;
        }
        if (!dst->buf)
            return false;

        concat_mode = -1;
        if (total_height == (dst->height & (~0x1)) && src[0]->width == dst->width) {
            // Vertical concat
            concat_mode = 0;
            if (total_height != dst->height) {  // Need fix
                std::lock_guard<std::mutex> lock(debug_mutex);
                print(src, dst);
            }
        } else if (total_width == dst->width /*&& src[0]->height <= dst->height*/) {  // fake alignment issue
            // Horizontal concat
            concat_mode = 1;
            if (src[0]->height != dst->height || src[1]->height != dst->height) {  // Need fix
                std::lock_guard<std::mutex> lock(debug_mutex);
                print(src, dst);
                auto actual_height = std::min(src[0]->height, dst->height);
                actual_height = std::min(src[1]->height, actual_height);
                src[0]->height = src[1]->height = dst->height = actual_height;
            }
        } else {
            return false;
        }
        return true;
    }

    std::vector<cldnn::event::ptr> concat(cldnn::stream& stream,
                                          std::vector<std::shared_ptr<tensor_concat_memory>>& src,
                                          std::shared_ptr<tensor_concat_memory>& dst) {
        const auto start = perf_dump_start();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();

        if (!validate(src, dst)) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            print(src, dst);
            std::cout << "simple_tensor_concat::validate failed due to src/dst mismatch." << std::endl;
        }

        size_t src_rec[3] = {0, 0, 0};
        size_t dst_rec[3] = {0, 0, 0};
        std::vector<cldnn::event::ptr> sync_events;
        if (concat_mode == 0) {
            // Vertical concat
            cl_event event;
            for (size_t i = 0; i < src.size(); i++) {
                size_t rect[3] = {src[i]->width, src[i]->height, 1};
                auto ret = clEnqueueCopyBufferRect(queue,
                                                   src[i]->buf,
                                                   dst->buf,
                                                   src_rec,
                                                   dst_rec,
                                                   rect,
                                                   src[i]->stride,
                                                   src[i]->height * src[i]->stride,
                                                   dst->stride,
                                                   dst->stride * dst->width,
                                                   0,
                                                   nullptr,
                                                   &event);
                if (ret != CL_SUCCESS) {
                    std::cout << "0.clEnqueueCopyBufferRect failed: " << oclErrorCode[ret] << ", idx = " << i
                              << std::endl;
                    OPENVINO_THROW("0.clEnqueueCopyBufferRect failed: ", oclErrorCode[ret], ", idx = ", i);
                }
                dst_rec[1] += src[i]->height;
                // ret = clWaitForEvents(1, &event);
                // CHECK_OCL_ERROR(ret, "clWaitForEvents failed");
                // clReleaseEvent(event);
                sync_events.emplace_back(ocl_stream.create_event(cl::Event(event)));
            }
        } else if (concat_mode == 1) {
            // Horizontal concat
            cl_event event;
            for (size_t i = 0; i < src.size(); i++) {
                size_t rect[3] = {src[i]->width, src[i]->height, 1};
                auto ret = clEnqueueCopyBufferRect(queue,
                                                   src[i]->buf,
                                                   dst->buf,
                                                   src_rec,
                                                   dst_rec,
                                                   rect,
                                                   src[i]->stride,
                                                   src[i]->height * src[i]->stride,
                                                   dst->stride,
                                                   dst->stride * dst->width,
                                                   0,
                                                   nullptr,
                                                   &event);
                if (ret != CL_SUCCESS) {
                    std::cout << "1.clEnqueueCopyBufferRect failed: " << oclErrorCode[ret] << ", idx = " << i
                              << std::endl;
                    OPENVINO_THROW("1.clEnqueueCopyBufferRect failed: ", oclErrorCode[ret], ", idx = ", i);
                }
                dst_rec[0] += src[i]->width;
                // ret = clWaitForEvents(1, &event);
                // CHECK_OCL_ERROR(ret, "clWaitForEvents failed");
                // clReleaseEvent(event);
                sync_events.emplace_back(ocl_stream.create_event(cl::Event(event)));
            }
        } else {
            std::cout << "tensor_concat failed: incorrect concat mode!" << std::endl;
            OPENVINO_THROW("tensor_concat failed: incorrect concat mode!");
        }

        if (0) {
            for (auto& event : sync_events)
                event->wait();
            sync_events.clear();
        }

        perf_dump_done(start, std::string("tensor_concat"), true);
        // return sync_events.size() > 0 ? stream.group_events(sync_events) : stream.create_user_event(true);
        return sync_events;
    }

    void print(const std::vector<std::shared_ptr<tensor_concat_memory>>& src, const std::shared_ptr<tensor_concat_memory>& dst) {
        for (size_t i = 0; i < src.size(); i++) {
            std::cout << " src[" << i << "]: ";
            src[i]->print();
        }
        std::cout << " dst[0]: ";
        dst->print();
        std::cout << std::endl;
    }

private:
    // 0 - vertical concat
    // 1 - horizontal concat
    int concat_mode;
};

static gpu_p2p_helper& get_p2p_instance() {
    static gpu_p2p_helper gpu_p2p_instance;
    return gpu_p2p_instance;
}

static simple_tensor_add& get_adder_instance(size_t idx) {
    static simple_tensor_add adder_instance[4];
    return adder_instance[idx];
}

struct sync_tensor_impl : public typed_primitive_impl<sync_tensor> {
    using parent = typed_primitive_impl<sync_tensor>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::sync_tensor_impl)

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

    void wait_p2p_done(cldnn::stream& stream,
                       cldnn::ocl::gpu_p2p_helper& p2p_helper,
                       ov::intel_gpu::SubMemoryManager::ptr& sub_mem_mgr,
                       int id,
                       size_t w_size,
                       int32_t w_rank,
                       bool validate = true) {
        // Wait for P2P transferred data are ready
        std::vector<int> copy_list(w_size, 1);
        copy_list[w_rank] = 0;
        auto start = perf_dump_start();

        // Wait P2P done
        while (true) {
            for (size_t idx = 0; idx < w_size; idx++) {
                if (idx != static_cast<size_t>(w_rank) && copy_list[idx]) {
                    auto& remote_ocl_stream =
                        downcast<ocl::ocl_stream>(*sub_mem_mgr->_memorys_table[id][idx].stream_ptr);
                    auto event = sub_mem_mgr->_memorys_table[id][w_rank].events[idx];
                    if (event) {
                        event->wait();
                        remote_ocl_stream.finish();
                        copy_list[idx] = 0;
                        // std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                        sub_mem_mgr->_memorys_table[id][w_rank].events[idx] = nullptr;
                        // MUST release remote cl_mem, but it will cause remote map failed.
                        // cl_mem remote_mem =
                        // static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][idx].remote_mem[w_rank]);
                        // clReleaseMemObject(remote_mem); // MUST releas remote cl_mem to avoid OUT OF RESOURCE
                    }
                }
            }
            auto left_size = std::accumulate(copy_list.begin(), copy_list.end(), 0);
            if (left_size == 0)
                break;
            auto end = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end - start;
            if (duration.count() > 10000) {
                start = perf_dump_start();
                std::cout << "rank[" << w_rank << "]Error: sync_tensor wait_p2p_done timeout..." << std::endl;
            }
        }
        perf_dump_done(start,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait p2p done"),
                       true);
        return;
    }

    void print_internal_buffer(sync_tensor_inst& instance,
                               std::vector<cldnn::memory::ptr>& bufs,
                               cldnn::layout& layout,
                               size_t w_size,
                               size_t w_rank) {
        for (size_t i = 0; i < w_size; i++) {
            std::cout << "\trank[" << w_rank << "]: bufs[" << i << "] = " << bufs[i]
                      << ", layout = " << layout.to_short_string();
            if (bufs[i]) {
                auto cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(bufs[i])->get_buffer().get();
                size_t data_size = 0;
                auto err = clGetMemObjectInfo(cl_buf, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL);
                CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_SIZE failed");
                std::cout << ", buf_size = " << data_size;
            }
            std::cout << std::endl;
        }
    }

    void update_internal_buffer(sync_tensor_inst& instance,
                                std::vector<cldnn::memory::ptr>& bufs,
                                cldnn::layout& last_layout,
                                cldnn::layout& layout,
                                size_t w_size,
                                size_t w_rank) {
        auto& engine = instance.get_network().get_engine();
        size_t required_size = layout.bytes_count();
        if (0) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << "Before update_internal_buffer: " << std::endl;
            print_internal_buffer(instance, bufs, layout, w_size, w_rank);
        }

        auto need_realloc = [&](size_t idx) {
            if (idx == w_rank)
                return false;

            if (bufs[idx] == nullptr || last_layout.bytes_count() == 0)
                return true;

            if (bufs[idx]->size() < required_size)
                return true;

            // Batch has been changed to smaller, need reallocate to decrease memory.
            auto last_batch = last_layout.batch() * last_layout.feature();
            auto batch = layout.batch() * layout.feature();
            if (last_batch > batch)
                return true;
            return false;
        };
        for (size_t i = 0; i < w_size; i++) {
            if (!need_realloc(i)) {
                continue;
            }
            size_t origin_size = bufs[i] != nullptr ? bufs[i]->size() : 0;
            bufs[i] = nullptr;
            bufs[i] = engine.allocate_memory(layout, cldnn::allocation_type::cl_mem, false);
            if (debug_enable) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                std::cout << "tensor_sync allocate: rank[" << w_rank << "]: layout[" << i
                          << "]=" << layout.to_short_string() << ", required_size = " << required_size
                          << ", current_size = " << origin_size << ", to_allocate_size = " << bufs[i]->size()
                          << std::endl;
            }
        }
        if (0) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << "After update_internal_buffer: " << std::endl;
            print_internal_buffer(instance, bufs, layout, w_size, w_rank);
            std::cout << std::endl;
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& stream = instance.get_network().get_stream();
        const bool pass_through_events = false;

        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto is_all_reduce = instance.get_impl_params()->need_add == true;
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
                    for (size_t j = 0; j < w_size; j++)
                        sub_mem_mgr->_memorys_table[id][i].events[j] = nullptr;
                }
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
            auto end_1 = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end_1 - start_1;
            if (duration.count() > 10000) {
                start_1 = perf_dump_start();
                std::cout << "rank[" << w_rank << "]Error: sync_tensor wait data ready timeout..." << std::endl;
            }
        }

        perf_dump_done(start_1,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait data ready"),
                       true);
        gpu_p2p_helper& gpu_p2p_instance = get_p2p_instance();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto local_context = ocl_stream.get_engine().get_cl_context().get();
        auto p2p_src_layout = instance.get_output_layout(0);
        if (is_all_reduce) {
            OPENVINO_ASSERT(1 == instance.get_output_memorys().size(), "All reduce only has one output!");
            sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[w_rank] = instance.get_output_memorys()[0];
            // Allocate or reuse buffer for P2P target, same shape with output[0]
            p2p_src_layout = instance.get_output_layout(0);
            update_internal_buffer(instance,
                                   sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs,
                                   sub_mem_mgr->_memorys_table[id][w_rank].layout,
                                   p2p_src_layout,
                                   w_size,
                                   w_rank);
        } else {
            OPENVINO_ASSERT(2 == instance.get_output_memorys().size(),
                            "All gather need additional buffer for concat result!");
            sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[w_rank] = instance.get_output_memorys()[1];
            // Allocate or reuse buffer for P2P target, same shape with output[1]
            p2p_src_layout = instance.get_output_layout(1);
            update_internal_buffer(instance,
                                   sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs,
                                   sub_mem_mgr->_memorys_table[id][w_rank].layout,
                                   p2p_src_layout,
                                   w_size,
                                   w_rank);
        }
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
        auto src_p2p_buf =
            std::dynamic_pointer_cast<const ocl::gpu_buffer>(sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[w_rank]);
        auto src_cl_buf = src_p2p_buf->get_buffer().get();
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][idx].recv_bufs[w_rank];
                    auto dst_cl_buf_remote =
                        std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();

                    data_size = dst_mem->size();
                    auto dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                    auto p2p_data_size = p2p_src_layout.bytes_count();
                    if (w_size > 2) {
                        gpu_lock.acquire();
                        gpu_p2p_instance.remote_copy(stream, src_cl_buf, dst_cl_buf, p2p_data_size);
                        gpu_lock.signal();
                    } else {
                        gpu_p2p_instance.remote_copy(stream, src_cl_buf, dst_cl_buf, p2p_data_size);
                    }
                    // P2P has been done.
                    {
                        std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                        sub_mem_mgr->_memorys_table[id][idx].events[w_rank] = stream.create_user_event(true);
                    }
                    if (0) {
                        std::lock_guard<std::mutex> lock(debug_mutex);
                        printf("Write output memory (rank=%d):\n", w_rank);
                        dump_cl_buf(ocl_stream.get_cl_queue().get(), dst_cl_buf, dst_mem->count(), 0);
                    }
                    // gpu_p2p_instance.destory_remote_mem(dst_cl_buf);
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
                start_2 = perf_dump_start();
                std::cout << "rank[" << w_rank << "]Error: sync_tensor p2p write timeout..." << std::endl;
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

        // P2P adopts sync write to avoid the problem of event cannot work across contexts
        wait_p2p_done(stream, gpu_p2p_instance, sub_mem_mgr, id, w_size, w_rank, false);

        std::vector<cldnn::event::ptr> sync_events;
        if (is_all_reduce) {
            // All_reduce path
            auto start_3 = perf_dump_start();
            auto dst_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(0));
            auto dst_cl_buf = dst_mem->get_buffer().get();
            auto& adder_instance = get_adder_instance(w_rank);
            // auto data_size = dst_mem->size();
            for (size_t idx = 0; idx < w_size; idx++) {
                if (idx != static_cast<size_t>(w_rank)) {
                    auto src_mem = sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[idx];
                    auto src_cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(src_mem)->get_buffer().get();
                    sync_event = adder_instance.tensor_add(
                        stream,
                        src_cl_buf,
                        dst_cl_buf,
                        dst_mem->count(),
                        adder_instance.element_type_to_kernel_data_type(dst_mem->get_layout().data_type));
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
            // All_gather path: do concat
            auto concat = std::make_shared<simple_tensor_concat>();
            std::vector<std::shared_ptr<tensor_concat_memory>> src_mem;
            auto output_layout = instance.get_output_layout(0);
            ov::element::Type element_type = output_layout.data_type;
            auto element_size = element_type.size();
            for (size_t idx = 0; idx < w_size; idx++) {
                // auto mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(idx));
                auto mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                    sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[idx]);
                auto cl_buf = mem->get_buffer().get();
                auto layout = instance.get_output_layout(1);
                auto shape = layout.get_shape();
                auto width = shape[-1] * element_size;
                auto stride = shape[-1] * element_size;  // Need no pad?
                auto height = ov::shape_size(shape) / shape[-1];
                if (0) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    std::cout << "tensor_sync concat: rank[" << w_rank << "]: layout[" << idx << "] (" << height << ","
                              << width / element_size << ") = " << layout.to_short_string()
                              << ", offset = " << layout.get_linear_offset()
                              << ", linear size = " << layout.get_linear_size()
                              << ", buf_size = " << layout.get_buffer_size() << ", pitch[] = ";
                    auto pitches = layout.get_pitches().sizes();
                    for (auto& p : pitches) {
                        std::cout << p << ", ";
                    }
                    std::cout << "]" << std::endl;
                }

                auto _src = std::make_shared<tensor_concat_memory>(cl_buf, width, height, stride, element_type);
                src_mem.emplace_back(_src);
            }

            auto _dst_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(0));
            auto dst_cl_buf = _dst_mem->get_buffer().get();
            auto dst_shape = output_layout.get_shape();
            auto dst_width = dst_shape[-1] * element_size;
            auto dst_stride = dst_shape[-1] * element_size;  // Need no pad?
            auto dst_height = ov::shape_size(dst_shape) / dst_shape[-1];
            auto dst_mem = std::make_shared<tensor_concat_memory>(dst_cl_buf, dst_width, dst_height, dst_stride, element_type);

            if (0) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                concat->print(src_mem, dst_mem);
            }
            sync_events = concat->concat(stream, src_mem, dst_mem);
            if (0) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                auto element_size = dst_mem->type.size();
                for (size_t i = 0; i < src_mem.size(); i++) {
                    printf("Concat input memory %ld (rank=%d):\n", i, w_rank);
                    dump_cl_buf(ocl_stream.get_cl_queue().get(),
                                src_mem[i]->buf,
                                src_mem[i]->width / element_size * src_mem[i]->height,
                                0);
                }
                printf("Concat output memory (rank=%d):\n", w_rank);
                dump_cl_buf(ocl_stream.get_cl_queue().get(),
                            dst_mem->buf,
                            dst_mem->width / element_size * dst_mem->height,
                            0);
            }
        }

        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }
        perf_dump_done(start, std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor total"), true);

        // for (auto& evt : sync_events) {
        //     evt->wait();
        // }

        // This block MUST be put exactly at the end of this method.
        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_memorys_table[id][w_rank].layout = p2p_src_layout;
            sub_mem_mgr->_use_count[id]++;
        }

        return sync_events.size() > 0 ? stream.group_events(sync_events) : stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const sync_tensor_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<sync_tensor_impl>();
    }
};

namespace detail {

attach_sync_tensor_impl::attach_sync_tensor_impl() {
    implementation_map<sync_tensor>::add(impl_types::ocl, shape_types::dynamic_shape, sync_tensor_impl::create, {});
    implementation_map<sync_tensor>::add(impl_types::ocl, shape_types::static_shape, sync_tensor_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)
