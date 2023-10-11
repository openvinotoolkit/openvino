// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * OpenCL context and OpenCL shared memory blobs
 *
 * @file gpu_context_api_ocl.hpp
 */
#pragma once

// TODO: Remove after migration to new API in the benchmark app
#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <ie_remote_context.hpp>
#include <memory>
#include <string>

#include "gpu/details/gpu_context_helpers.hpp"
#include "gpu/gpu_ocl_wrapper.hpp"
#include "gpu/gpu_params.hpp"
#include "ie_compound_blob.h"
#include "ie_core.hpp"

namespace InferenceEngine {

namespace gpu {
/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with OpenCL context object.
 * The plugin object derived from this class can be obtained either with
 * GetContext() method of Executable network or using CreateContext() Core call.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED ClContext : public RemoteContext, public details::param_map_obj_getter {
public:
    /**
     * @brief A smart pointer to the ClContext object
     */
    using Ptr = std::shared_ptr<ClContext>;

    /**
     * @brief Returns the underlying OpenCL context handle.
     * @return `cl_context`
     */
    cl_context get() {
        return _ObjFromParams<cl_context, gpu_handle_param>(getParams(),
                                                            GPU_PARAM_KEY(OCL_CONTEXT),
                                                            GPU_PARAM_KEY(CONTEXT_TYPE),
                                                            GPU_PARAM_VALUE(OCL),
                                                            GPU_PARAM_VALUE(VA_SHARED));
    }

    /**
     * @brief OpenCL context handle conversion operator for the ClContext object.
     * @return `cl_context`
     */
    operator cl_context() {
        return get();
    }

    /**
     * @brief Standard Khronos cl::Context wrapper conversion operator for the ClContext object.
     * @return `cl::Context` object
     */
    operator cl::Context() {
        return cl::Context(get(), true);
    }
};

/**
 * @brief The basic class for all GPU plugin remote blob objects.
 * The OpenCL memory object handle (cl_mem) can be obtained from this class object.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED ClBlob : public RemoteBlob {
public:
    /**
     * @brief A smart pointer to the ClBlob object
     */
    using Ptr = std::shared_ptr<ClBlob>;

    /**
     * @brief Creates a ClBlob object with the specified dimensions and layout.
     * @param tensorDesc Tensor description
     */
    explicit ClBlob(const TensorDesc& tensorDesc) : RemoteBlob(tensorDesc) {}
};

/**
 * @brief This class represents an abstraction for GPU plugin remote blob
 * which can be shared with user-supplied OpenCL buffer.
 * The plugin object derived from this class can be obtained with CreateBlob() call.
 * @note User can obtain OpenCL buffer handle from this class.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED ClBufferBlob : public ClBlob, public details::param_map_obj_getter {
public:
    /**
     * @brief A smart pointer to the ClBufferBlob object
     */
    using Ptr = std::shared_ptr<ClBufferBlob>;

    /**
     * @brief Creates a ClBufferBlob object with the specified dimensions and layout.
     * @param tensorDesc Tensor description
     */
    explicit ClBufferBlob(const TensorDesc& tensorDesc) : ClBlob(tensorDesc) {}

    /**
     * @brief Returns the underlying OpenCL memory object handle.
     * @return underlying OpenCL memory object handle
     */
    cl_mem get() {
        return _ObjFromParams<cl_mem, gpu_handle_param>(getParams(),
                                                        GPU_PARAM_KEY(MEM_HANDLE),
                                                        GPU_PARAM_KEY(SHARED_MEM_TYPE),
                                                        GPU_PARAM_VALUE(OCL_BUFFER),
                                                        GPU_PARAM_VALUE(DX_BUFFER));
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
 * @brief This class represents an abstraction for GPU plugin remote blob
 * which can be shared with user-supplied USM pointer.
 * The plugin object derived from this class can be obtained with CreateBlob() call.
 * @note User can obtain USM pointer from this class.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED USMBlob : public ClBlob, public details::param_map_obj_getter {
public:
    /**
     * @brief A smart pointer to the ClBufferBlob object
     */
    using Ptr = std::shared_ptr<USMBlob>;

    /**
     * @brief Creates a ClBufferBlob object with the specified dimensions and layout.
     * @param tensorDesc Tensor description
     */
    explicit USMBlob(const TensorDesc& tensorDesc) : ClBlob(tensorDesc) {}

    /**
     * @brief Returns the underlying OpenCL memory object handle.
     * @return underlying OpenCL memory object handle
     */
    void* get() {
        const auto& params = getParams();
        auto itrType = params.find(GPU_PARAM_KEY(SHARED_MEM_TYPE));
        if (itrType == params.end())
            IE_THROW() << "Parameter of type " << GPU_PARAM_KEY(SHARED_MEM_TYPE) << " not found";

        auto mem_type = itrType->second.as<std::string>();
        if (mem_type != GPU_PARAM_VALUE(USM_USER_BUFFER) && mem_type != GPU_PARAM_VALUE(USM_HOST_BUFFER) &&
            mem_type != GPU_PARAM_VALUE(USM_DEVICE_BUFFER))
            IE_THROW() << "Unexpected USM blob type: " << mem_type;

        auto itrHandle = params.find(GPU_PARAM_KEY(MEM_HANDLE));
        if (itrHandle == params.end()) {
            IE_THROW() << "No parameter " << GPU_PARAM_KEY(MEM_HANDLE) << " found";
        }

        return itrHandle->second.as<gpu_handle_param>();
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote blob
 * which can be shared with user-supplied OpenCL 2D Image.
 * The plugin object derived from this class can be obtained with CreateBlob() call.
 * @note User can obtain OpenCL image handle from this class.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED ClImage2DBlob : public ClBlob, public details::param_map_obj_getter {
public:
    /**
     * @brief A smart pointer to the ClImage2DBlob object
     */
    using Ptr = std::shared_ptr<ClImage2DBlob>;

    /**
     * @brief Creates a ClImage2DBlob object with the specified dimensions and layout.
     * @param tensorDesc Tensor description
     */
    explicit ClImage2DBlob(const TensorDesc& tensorDesc) : ClBlob(tensorDesc) {}

    /**
     * @brief Returns the underlying OpenCL memory object handle.
     * @return `cl_mem`
     */
    cl_mem get() {
        return _ObjFromParams<cl_mem, gpu_handle_param>(getParams(),
                                                        GPU_PARAM_KEY(MEM_HANDLE),
                                                        GPU_PARAM_KEY(SHARED_MEM_TYPE),
                                                        GPU_PARAM_VALUE(OCL_IMAGE2D),
                                                        GPU_PARAM_VALUE(VA_SURFACE));
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
 * @brief This function is used to obtain remote context object from user-supplied OpenCL context handle
 * @param core A reference to Inference Engine Core object
 * @param deviceName A name of device to create a remote context for
 * @param ctx A OpenCL context to be used to create shared remote context
 * @param target_tile_id Desired tile id within given context for multi-tile system. Default value (-1) means that root
 * device should be used
 * @return A shared remote context instance
 */
INFERENCE_ENGINE_1_0_DEPRECATED static inline RemoteContext::Ptr make_shared_context(Core& core,
                                                                                     std::string deviceName,
                                                                                     cl_context ctx,
                                                                                     int target_tile_id = -1) {
    ParamMap contextParams = {{GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(OCL)},
                              {GPU_PARAM_KEY(OCL_CONTEXT), static_cast<gpu_handle_param>(ctx)},
                              {GPU_PARAM_KEY(TILE_ID), target_tile_id}};
    return core.CreateContext(deviceName, contextParams);
}

/**
 * @brief This function is used to obtain remote context object from user-supplied OpenCL context handle
 * @param core A reference to Inference Engine Core object
 * @param deviceName A name of device to create a remote context for
 * @param queue An OpenCL queue to be used to create shared remote context. Queue will be reused inside the plugin.
 * @note Only latency mode is supported for such context sharing case.
 * @return A shared remote context instance
 */
INFERENCE_ENGINE_1_0_DEPRECATED static inline RemoteContext::Ptr make_shared_context(Core& core,
                                                                                     std::string deviceName,
                                                                                     cl_command_queue queue) {
    cl_context ctx;
    auto res = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
    if (res != CL_SUCCESS)
        IE_THROW() << "Can't get context from given opencl queue";
    ParamMap contextParams = {{GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(OCL)},
                              {GPU_PARAM_KEY(OCL_CONTEXT), static_cast<gpu_handle_param>(ctx)},
                              {GPU_PARAM_KEY(OCL_QUEUE), static_cast<gpu_handle_param>(queue)}};
    return core.CreateContext(deviceName, contextParams);
}

/**
 * @brief This function is used to create remote blob object within default GPU plugin OpenCL context
 * @param desc A tensor descriptor object representing remote blob configuration
 * @param ctx A remote context used to create remote blob
 * @return A remote blob instance
 */
INFERENCE_ENGINE_1_0_DEPRECATED static inline Blob::Ptr make_shared_blob(const TensorDesc& desc, ClContext::Ptr ctx) {
    return std::dynamic_pointer_cast<Blob>(ctx->CreateBlob(desc));
}

/**
 * @brief This function is used to obtain remote blob object from user-supplied cl::Buffer wrapper object
 * @param desc A tensor descriptor object representing remote blob configuration
 * @param ctx A remote context used to create remote blob
 * @param buffer A cl::Buffer object wrapped by a remote blob
 * @return A remote blob instance
 */
INFERENCE_ENGINE_1_0_DEPRECATED static inline Blob::Ptr make_shared_blob(const TensorDesc& desc,
                                                                         RemoteContext::Ptr ctx,
                                                                         cl::Buffer& buffer) {
    auto casted = ctx->as<ClContext>();
    if (nullptr == casted) {
        IE_THROW() << "Invalid remote context passed";
    }

    ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_BUFFER)},
                       {GPU_PARAM_KEY(MEM_HANDLE), static_cast<gpu_handle_param>(buffer.get())}};
    return std::dynamic_pointer_cast<Blob>(casted->CreateBlob(desc, params));
}

/**
 * @brief This function is used to obtain remote blob object from user-supplied OpenCL buffer handle
 * @param desc A tensor descriptor object representing remote blob configuration
 * @param ctx A remote context used to create remote blob
 * @param buffer A cl_mem object wrapped by a remote blob
 * @return A remote blob instance
 */
INFERENCE_ENGINE_1_0_DEPRECATED static inline Blob::Ptr make_shared_blob(const TensorDesc& desc,
                                                                         RemoteContext::Ptr ctx,
                                                                         cl_mem buffer) {
    auto casted = ctx->as<ClContext>();
    if (nullptr == casted) {
        IE_THROW() << "Invalid remote context passed";
    }

    ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_BUFFER)},
                       {GPU_PARAM_KEY(MEM_HANDLE), static_cast<gpu_handle_param>(buffer)}};
    return std::dynamic_pointer_cast<Blob>(casted->CreateBlob(desc, params));
}

/**
 * @brief This function is used to obtain remote blob object from user-supplied cl::Image2D wrapper object
 * @param desc A tensor descriptor object representing remote blob configuration
 * @param ctx A remote context used to create remote blob
 * @param image A cl::Image2D object wrapped by a remote blob
 * @return A remote blob instance
 */
INFERENCE_ENGINE_1_0_DEPRECATED static inline Blob::Ptr make_shared_blob(const TensorDesc& desc,
                                                                         RemoteContext::Ptr ctx,
                                                                         cl::Image2D& image) {
    auto casted = ctx->as<ClContext>();
    if (nullptr == casted) {
        IE_THROW() << "Invalid remote context passed";
    }

    ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_IMAGE2D)},
                       {GPU_PARAM_KEY(MEM_HANDLE), static_cast<gpu_handle_param>(image.get())}};
    return std::dynamic_pointer_cast<Blob>(casted->CreateBlob(desc, params));
}

}  // namespace gpu

}  // namespace InferenceEngine

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif
