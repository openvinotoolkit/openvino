// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * shared Video Acceleration device contexts
 * and shared memory blobs which contain Video Acceleration surfaces
 *
 * @file gpu_context_api_va.hpp
 */
#pragma once

#include <memory>
#include <string>

#include "gpu/gpu_context_api_ocl.hpp"

#include <va/va.h>

namespace InferenceEngine {

namespace gpu {
/**
* @brief This class represents an abstraction for GPU plugin remote context
* which is shared with VA display object.
* The plugin object derived from this class can be obtained either with
* GetContext() method of Executable network or using CreateContext() Core call.
* @note User can also obtain OpenCL context handle from this class.
*/
class VAContext : public ClContext {
public:
    /**
    * @brief A smart pointer to the VAContext object
    */
    using Ptr = std::shared_ptr<VAContext>;

    /**
     * @brief VADisplay conversion operator for the VAContext object.
     * @return Underlying VADisplay object handle 
     */
    operator VADisplay() {
        return _ObjFromParams<VADisplay, gpu_handle_param>(getParams(),
            GPU_PARAM_KEY(VA_DEVICE),
            GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(VA_SHARED));
    }
};

/**
* @brief This class represents an abstraction for GPU plugin remote blob
* which is shared with VA output surface.
* The plugin object derived from this class can be obtained with CreateBlob() call.
* @note User can also obtain OpenCL 2D image handle from this class.
*/
class VASurfaceBlob : public ClImage2DBlob {
public:
    /**
    * @brief A smart pointer to the VASurfaceBlob object
    */
    using Ptr = std::shared_ptr<VASurfaceBlob>;

    /**
     * @brief Creates a VASurfaceBlob object with the specified dimensions and layout.
     * @param tensorDesc Tensor description
     */
    explicit VASurfaceBlob(const TensorDesc& tensorDesc) : ClImage2DBlob(tensorDesc) {}

    /**
     * @brief VASurfaceID conversion operator for the VASurfaceBlob object.
     * @return VA surface handle 
     */
    operator VASurfaceID() {
        return _ObjFromParams<VASurfaceID, uint32_t>(getParams(),
            GPU_PARAM_KEY(DEV_OBJECT_HANDLE),
            GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE));
    }

    /**
     * @brief Returns plane ID of underlying video decoder surface
     */
    uint32_t plane() {
        return _ObjFromParams<uint32_t, uint32_t>(getParams(),
            GPU_PARAM_KEY(VA_PLANE),
            GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE));
    }
};

/**
* @brief This function is used to obtain a NV12 compound blob object from NV12 VA decoder output.
* The resulting compound contains two remote blobs for Y and UV planes of the surface.
*/
static inline Blob::Ptr make_shared_blob_nv12(size_t height, size_t width, RemoteContext::Ptr ctx, VASurfaceID nv12_surf) {
    auto casted = std::dynamic_pointer_cast<VAContext>(ctx);
    if (nullptr == casted) {
        THROW_IE_EXCEPTION << "Invalid remote context passed";
    }

    // despite of layout, blob dimensions always follow in N,C,H,W order
    TensorDesc ydesc(Precision::U8, { 1, 1, height, width }, Layout::NHWC);
    ParamMap blobParams = {
        { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE) },
        { GPU_PARAM_KEY(DEV_OBJECT_HANDLE), nv12_surf },
        { GPU_PARAM_KEY(VA_PLANE), uint32_t(0) }
    };
    Blob::Ptr y_blob = std::dynamic_pointer_cast<Blob>(casted->CreateBlob(ydesc, blobParams));

    TensorDesc uvdesc(Precision::U8, { 1, 2, height / 2, width / 2 }, Layout::NHWC);
    blobParams[GPU_PARAM_KEY(VA_PLANE)] = uint32_t(1);
    Blob::Ptr uv_blob = std::dynamic_pointer_cast<Blob>(casted->CreateBlob(uvdesc, blobParams));

    return InferenceEngine::make_shared_blob<NV12Blob>(y_blob, uv_blob);
}

/**
* @brief This function is used to obtain remote context object from VA display handle
*/
static inline VAContext::Ptr make_shared_context(Core& core, std::string deviceName, VADisplay device) {
    ParamMap contextParams = {
        { GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(VA_SHARED) },
        { GPU_PARAM_KEY(VA_DEVICE), static_cast<gpu_handle_param>(device) }
    };
    return std::dynamic_pointer_cast<VAContext>(core.CreateContext(deviceName, contextParams));
}

/**
* @brief This function is used to obtain remote blob object from VA surface handle
*/
static inline VASurfaceBlob::Ptr make_shared_blob(const TensorDesc& desc, RemoteContext::Ptr ctx, VASurfaceID surface, uint32_t plane = 0) {
    auto casted = std::dynamic_pointer_cast<VAContext>(ctx);
    if (nullptr == casted) {
        THROW_IE_EXCEPTION << "Invalid remote context passed";
    }
    ParamMap params = {
        { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE) },
        { GPU_PARAM_KEY(DEV_OBJECT_HANDLE), surface },
        { GPU_PARAM_KEY(VA_PLANE), plane }
    };
    return std::dynamic_pointer_cast<VASurfaceBlob>(casted->CreateBlob(desc, params));
}

}  // namespace gpu
}  // namespace InferenceEngine
