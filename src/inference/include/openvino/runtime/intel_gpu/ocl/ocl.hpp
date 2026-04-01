// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * OpenCL context and OpenCL shared memory tensors
 *
 * @file openvino/runtime/intel_gpu/ocl/ocl.hpp
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <CL/cl_ext.h>

#ifndef CL_DEVICE_HANDLE_LIST_KHR
#define CL_DEVICE_HANDLE_LIST_KHR 0x2051
#endif

#ifndef CL_DEVICE_HANDLE_LIST_END_KHR
#define CL_DEVICE_HANDLE_LIST_END_KHR 0
#endif

#ifndef CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR
#define CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR 0x2062
#endif

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @defgroup ov_runtime_ocl_gpu_cpp_api Intel GPU OpenCL interoperability
 * @ingroup ov_runtime_cpp_api
 * Set of C++ classes and properties to work with Remote API for Intel GPU OpenCL plugin.
 */

/**
 * @brief Namespace with Intel GPU OpenCL specific remote objects
 */
namespace ocl {

/**
 * @brief Shortcut for defining a handle parameter
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
using gpu_handle_param = void*;

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which can be shared with user-supplied OpenCL buffer.
 * The plugin object derived from this class can be obtained with ClContext::create_tensor() call.
 * @note User can obtain OpenCL buffer handle from this class.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class ClBufferTensor : public RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor,
                                 {{std::string(ov::intel_gpu::mem_handle.name()), {}},
                                  {std::string(ov::intel_gpu::shared_mem_type.name()),
                                   {ov::Any(ov::intel_gpu::SharedMemType::OCL_BUFFER).as<std::string>(),
                                    ov::Any(ov::intel_gpu::SharedMemType::DX_BUFFER).as<std::string>()}}});
    }

    /**
     * @brief Returns the underlying OpenCL memory object handle.
     * @return underlying OpenCL memory object handle
     */
    cl_mem get() {
        return static_cast<cl_mem>(get_params().at(ov::intel_gpu::mem_handle.name()).as<gpu_handle_param>());
    }

    /**
     * @brief OpenCL memory handle conversion operator.
     * @return `cl_mem`
     */
    operator cl_mem() {
        return get();
    }

    /**
     * @brief Standard Khronos cl::Buffer wrapper conversion operator.
     * @return `cl::Buffer` object
     */
    operator cl::Buffer() {
        return cl::Buffer(get(), true);
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which can be shared with user-supplied OpenCL 2D Image.
 * The plugin object derived from this class can be obtained with ClContext::create_tensor() call.
 * @note User can obtain OpenCL image handle from this class.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class ClImage2DTensor : public RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor,
                                 {{std::string(ov::intel_gpu::mem_handle.name()), {}},
                                  {std::string(ov::intel_gpu::shared_mem_type.name()),
                                   {ov::Any(ov::intel_gpu::SharedMemType::OCL_IMAGE2D).as<std::string>(),
                                    ov::Any(ov::intel_gpu::SharedMemType::VA_SURFACE).as<std::string>()}}});
    }

    /**
     * @brief Returns the underlying OpenCL memory object handle.
     * @return underlying OpenCL memory object handle
     */
    cl_mem get() {
        return static_cast<cl_mem>(get_params().at(ov::intel_gpu::mem_handle.name()).as<gpu_handle_param>());
    }

    /**
     * @brief OpenCL memory handle conversion operator.
     * @return `cl_mem`
     */
    operator cl_mem() {
        return get();
    }

    /**
     * @brief Standard Khronos cl::Image2D wrapper conversion operator for the ClContext object.
     * @return `cl::Image2D` object
     */
    operator cl::Image2D() {
        return cl::Image2D(get(), true);
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which can be shared with user-supplied USM device pointer.
 * The plugin object derived from this class can be obtained with ClContext::create_tensor() call.
 * @note User can obtain USM pointer from this class.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class USMTensor : public RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor,
                                 {{std::string(ov::intel_gpu::mem_handle.name()), {}},
                                  {std::string(ov::intel_gpu::shared_mem_type.name()),
                                   {ov::Any(ov::intel_gpu::SharedMemType::USM_USER_BUFFER).as<std::string>(),
                                    ov::Any(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER).as<std::string>(),
                                    ov::Any(ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER).as<std::string>()}}});
    }

    /**
     * @brief Returns the underlying USM pointer.
     * @return underlying USM pointer
     */
    void* get() {
        return static_cast<void*>(get_params().at(ov::intel_gpu::mem_handle.name()).as<void*>());
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with OpenCL context object.
 * The plugin object derived from this class can be obtained either with
 * CompiledModel::get_context() or Core::create_context() calls.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
class ClContext : public RemoteContext {
protected:
    /**
     * @brief GPU device name
     */
    static constexpr const char* device_name = "GPU";

    /**
     * @brief Default constructor which can be used in derived classes to avoid multiple create_context() calls
     */
    ClContext() = default;

public:
    // Needed to make create_tensor overloads from base class visible for user
    using RemoteContext::create_tensor;
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param remote_context A remote context to check
     */
    static void type_check(const RemoteContext& remote_context) {
        RemoteContext::type_check(remote_context,
                                  {{std::string(ov::intel_gpu::ocl_context.name()), {}},
                                   {std::string(ov::intel_gpu::context_type.name()),
                                    {ov::Any(ov::intel_gpu::ContextType::OCL).as<std::string>(),
                                     ov::Any(ov::intel_gpu::ContextType::VA_SHARED).as<std::string>()}}});
    }

    /**
     * @brief Constructs context object from user-supplied OpenCL context handle
     * @param core A reference to OpenVINO Runtime Core object
     * @param ctx A OpenCL context to be used to create shared remote context
     * @param ctx_device_id An ID of device to be used from ctx
     */
    ClContext(Core& core, cl_context ctx, int ctx_device_id = 0) {
        AnyMap context_params = {{ov::intel_gpu::context_type.name(), ov::intel_gpu::ContextType::OCL},
                                 {ov::intel_gpu::ocl_context.name(), static_cast<gpu_handle_param>(ctx)},
                                 {ov::intel_gpu::ocl_context_device_id.name(), ctx_device_id}};
        *this = core.create_context(device_name, context_params).as<ClContext>();
    }

    /**
     * @brief Constructs context object from user-supplied OpenCL context handle
     * @param core A reference to OpenVINO Runtime Core object
     * @param queue An OpenCL queue to be used to create shared remote context. Queue will be reused inside the plugin.
     * @note Only latency mode is supported for such context sharing case.
     */
    ClContext(Core& core, cl_command_queue queue) {
        cl_context ctx;
        auto res = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
        OPENVINO_ASSERT(res == CL_SUCCESS, "Can't get context from given opencl queue");
        AnyMap context_params = {{ov::intel_gpu::context_type.name(), ov::intel_gpu::ContextType::OCL},
                                 {ov::intel_gpu::ocl_context.name(), static_cast<gpu_handle_param>(ctx)},
                                 {ov::intel_gpu::ocl_queue.name(), static_cast<gpu_handle_param>(queue)}};
        *this = core.create_context(device_name, context_params).as<ClContext>();
    }

    /**
     * @brief Returns the underlying OpenCL context handle.
     * @return `cl_context`
     */
    cl_context get() {
        return static_cast<cl_context>(get_params().at(ov::intel_gpu::ocl_context.name()).as<gpu_handle_param>());
    }

    cl_context get() const {
        return static_cast<cl_context>(get_params().at(ov::intel_gpu::ocl_context.name()).as<gpu_handle_param>());
    }

    /**
     * @brief OpenCL context handle conversion operator for the ClContext object.
     * @return `cl_context`
     */
    operator cl_context() {
        return get();
    }

    operator cl_context() const {
        return get();
    }

    /**
     * @brief Standard Khronos cl::Context wrapper conversion operator for the ClContext object.
     * @return `cl::Context` object
     */
    operator cl::Context() {
        return cl::Context(get(), true);
    }

    /**
     * @brief This function is used to construct a NV12 compound tensor object from two cl::Image2D wrapper objects.
     * The resulting compound contains two remote tensors for Y and UV planes of the surface.
     * @param nv12_image_plane_y cl::Image2D object containing Y plane data.
     * @param nv12_image_plane_uv cl::Image2D object containing UV plane data.
     * @return A pair of remote tensors for each plane
     */
    std::pair<ClImage2DTensor, ClImage2DTensor> create_tensor_nv12(const cl::Image2D& nv12_image_plane_y,
                                                                   const cl::Image2D& nv12_image_plane_uv) {
        size_t width = nv12_image_plane_y.getImageInfo<CL_IMAGE_WIDTH>();
        size_t height = nv12_image_plane_y.getImageInfo<CL_IMAGE_HEIGHT>();
        AnyMap tensor_params = {
            {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::OCL_IMAGE2D},
            {ov::intel_gpu::mem_handle.name(), static_cast<gpu_handle_param>(nv12_image_plane_y.get())}};
        auto y_tensor = create_tensor(element::u8, {1, height, width, 1}, tensor_params);
        tensor_params[ov::intel_gpu::mem_handle.name()] = static_cast<gpu_handle_param>(nv12_image_plane_uv.get());
        auto uv_tensor = create_tensor(element::u8, {1, height / 2, width / 2, 2}, tensor_params);
        return std::make_pair(y_tensor.as<ClImage2DTensor>(), uv_tensor.as<ClImage2DTensor>());
    }

    /**
     * @brief This function is used to obtain remote tensor object from user-supplied cl_mem object
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param buffer A cl_mem object that should be wrapped by a remote tensor
     * @return A remote tensor instance
     */
    ClBufferTensor create_tensor(const element::Type type, const Shape& shape, const cl_mem buffer) {
        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::OCL_BUFFER},
                         {ov::intel_gpu::mem_handle.name(), static_cast<gpu_handle_param>(buffer)}};
        return create_tensor(type, shape, params).as<ClBufferTensor>();
    }

    /**
     * @brief This function is used to obtain remote tensor object from user-supplied cl::Buffer object
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param buffer A cl::Buffer object that should be wrapped by a remote tensor
     * @return A remote tensor instance
     */
    ClBufferTensor create_tensor(const element::Type type, const Shape& shape, const cl::Buffer& buffer) {
        return create_tensor(type, shape, buffer.get());
    }

    /**
     * @brief This function is used to obtain remote tensor object from user-supplied cl::Image2D object
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param image A cl::Image2D object that should be wrapped by a remote tensor
     * @return A remote tensor instance
     */
    ClImage2DTensor create_tensor(const element::Type type, const Shape& shape, const cl::Image2D& image) {
        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::OCL_IMAGE2D},
                         {ov::intel_gpu::mem_handle.name(), static_cast<gpu_handle_param>(image.get())}};
        return create_tensor(type, shape, params).as<ClImage2DTensor>();
    }

    /**
     * @brief This function is used to obtain remote tensor object from user-supplied shared OpenCL buffer handle.
     *        The API mirrors the NPU pointer-based create_tensor form.
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param shared_buffer A shared OpenCL buffer handle passed as void*
     * @param memory_type Memory type to use (default: SHARED_BUF)
     * @note CPU_VA memory type is currently not supported in GPU OCL context API.
     *       For CPU virtual address allocations, pointer and allocation size must be aligned to 4KB,
     *       and allocation lifetime must outlive all infer requests and remote tensor lifetime.
     * @return A remote tensor instance
     */
    ClBufferTensor create_tensor(const element::Type type,
                                 const Shape& shape,
                                 void* shared_buffer,
                                 const MemType memory_type) {
        OPENVINO_ASSERT(memory_type == MemType::SHARED_BUF,
                        "Only SHARED_BUF memory type is currently supported for GPU shared_buffer API");
        OPENVINO_ASSERT(shared_buffer != nullptr,
                        "shared_buffer must not be nullptr for SHARED_BUF memory type");

        size_t byte_size = type.size();
        for (const auto& dim : shape) {
            byte_size *= dim;
        }

        // External-memory import needs OpenCL 3.0 buffer-properties API in headers.
#if defined(CL_VERSION_3_0)
        cl_int errcode_ret = CL_SUCCESS;
        const auto cl_ctx = static_cast<cl_context>(get_params().at(ov::intel_gpu::ocl_context.name()).as<gpu_handle_param>());

        size_t devices_size = 0;
        errcode_ret = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, 0, nullptr, &devices_size);
        OPENVINO_ASSERT(errcode_ret == CL_SUCCESS && devices_size >= sizeof(cl_device_id),
                "Failed to query OpenCL context devices, error code: ",
                errcode_ret);

        std::vector<cl_device_id> devices(devices_size / sizeof(cl_device_id));
        errcode_ret = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, devices_size, devices.data(), nullptr);
        OPENVINO_ASSERT(errcode_ret == CL_SUCCESS && !devices.empty(),
                "Failed to get OpenCL context devices, error code: ",
                errcode_ret);

        const auto device_id = devices.front();

        auto try_import_external_mem = [&](cl_mem_properties handle_type) -> cl_mem {
            const cl_mem_properties ext_mem_properties[] = {
            handle_type,
            static_cast<cl_mem_properties>(reinterpret_cast<intptr_t>(shared_buffer)),
            static_cast<cl_mem_properties>(CL_DEVICE_HANDLE_LIST_KHR),
            static_cast<cl_mem_properties>(reinterpret_cast<intptr_t>(device_id)),
            static_cast<cl_mem_properties>(CL_DEVICE_HANDLE_LIST_END_KHR),
            0
            };

            return clCreateBufferWithProperties(cl_ctx,
                            ext_mem_properties,
                            CL_MEM_READ_WRITE,
                            byte_size,
                            nullptr,
                            &errcode_ret);
        };

        cl_mem ext_mem_buffer = nullptr;
    #ifdef _WIN32
        // Win32 sharing can expose either NT or KMT handles depending on DXGI sharing mode.
        ext_mem_buffer = try_import_external_mem(static_cast<cl_mem_properties>(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR));
        if ((errcode_ret != CL_SUCCESS || ext_mem_buffer == nullptr)) {
            ext_mem_buffer = try_import_external_mem(static_cast<cl_mem_properties>(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR));
        }
    #else
        ext_mem_buffer = try_import_external_mem(static_cast<cl_mem_properties>(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR));
    #endif

        if (errcode_ret == CL_SUCCESS && ext_mem_buffer != nullptr) {
            struct ClMemReleaser {
                void operator()(cl_mem mem_obj) const {
                    if (mem_obj != nullptr) {
                        clReleaseMemObject(mem_obj);
                    }
                }
            };

            std::unique_ptr<_cl_mem, ClMemReleaser> ext_mem_guard(ext_mem_buffer);
            return create_tensor(type, shape, ext_mem_buffer);
        }
#endif

        // Keep compatibility for existing callers that pass cl_mem wrapped as void*.
        return create_tensor(type, shape, static_cast<cl_mem>(shared_buffer));
    }

    /**
     * @brief This function is used to obtain remote tensor object from user-supplied USM pointer
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param usm_ptr A USM pointer that should be wrapped by a remote tensor
     * @return A remote tensor instance
     */
    USMTensor create_tensor(const element::Type type, const Shape& shape, void* usm_ptr) {
        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
                         {ov::intel_gpu::mem_handle.name(), static_cast<gpu_handle_param>(usm_ptr)}};
        return create_tensor(type, shape, params).as<USMTensor>();
    }

    /**
     * @brief This function is used to allocate USM tensor with host allocation type
     * @param type Tensor element type
     * @param shape Tensor shape
     * @return A remote tensor instance
     */
    USMTensor create_usm_host_tensor(const element::Type type, const Shape& shape) {
        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}};
        return create_tensor(type, shape, params).as<USMTensor>();
    }

    /**
     * @brief This function is used to allocate USM tensor with device allocation type
     * @param type Tensor element type
     * @param shape Tensor shape
     * @return A remote tensor instance
     */
    USMTensor create_usm_device_tensor(const element::Type type, const Shape& shape) {
        AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER}};
        return create_tensor(type, shape, params).as<USMTensor>();
    }
};

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
