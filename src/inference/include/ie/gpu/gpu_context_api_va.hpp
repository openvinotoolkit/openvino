// Copyright (C) 2018-2023 Intel Corporation
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

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <memory>
#include <string>

#include "gpu/gpu_context_api_ocl.hpp"

// clang-format off
#include <va/va.h>
// clang-format on

namespace InferenceEngine {

namespace gpu {
/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with VA display object.
 * The plugin object derived from this class can be obtained either with
 * GetContext() method of Executable network or using CreateContext() Core call.
 * @note User can also obtain OpenCL context handle from this class.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED VAContext : public ClContext {
public:
    /**
     * @brief A smart pointer to the VAContext object
     */
    using Ptr = std::shared_ptr<VAContext>;

    /**
     * @brief `VADisplay` conversion operator for the VAContext object.
     * @return Underlying `VADisplay` object handle
     */
    operator VADisplay() {
        return _ObjFromParams<VADisplay, gpu_handle_param>(getParams(),
                                                           GPU_PARAM_KEY(VA_DEVICE),
                                                           GPU_PARAM_KEY(CONTEXT_TYPE),
                                                           GPU_PARAM_VALUE(VA_SHARED));
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote blob
 * which is shared with VA output surface.
 * The plugin object derived from this class can be obtained with CreateBlob() call.
 * @note User can also obtain OpenCL 2D image handle from this class.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED VASurfaceBlob : public ClImage2DBlob {
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
     * @return `VASurfaceID` handle
     */
    operator VASurfaceID() {
        return _ObjFromParams<VASurfaceID, uint32_t>(getParams(),
                                                     GPU_PARAM_KEY(DEV_OBJECT_HANDLE),
                                                     GPU_PARAM_KEY(SHARED_MEM_TYPE),
                                                     GPU_PARAM_VALUE(VA_SURFACE));
    }

    /**
     * @brief Returns plane ID of underlying video decoder surface
     * @return Plane ID
     */
    uint32_t plane() {
        return _ObjFromParams<uint32_t, uint32_t>(getParams(),
                                                  GPU_PARAM_KEY(VA_PLANE),
                                                  GPU_PARAM_KEY(SHARED_MEM_TYPE),
                                                  GPU_PARAM_VALUE(VA_SURFACE));
    }
};

/**
 * @brief This function is used to obtain remote context object from VA display handle
 * @param core Inference Engine Core object
 * @param deviceName A device name to create a remote context for
 * @param device A `VADisplay` to create remote context from
 * @param target_tile_id Desired tile id within given context for multi-tile system. Default value (-1) means that root
 * device should be used
 * @return A remote context wrapping `VADisplay`
 */
INFERENCE_ENGINE_1_0_DEPRECATED static inline VAContext::Ptr make_shared_context(Core& core,
                                                                                 std::string deviceName,
                                                                                 VADisplay device,
                                                                                 int target_tile_id = -1) {
    ParamMap contextParams = {{GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(VA_SHARED)},
                              {GPU_PARAM_KEY(VA_DEVICE), static_cast<gpu_handle_param>(device)},
                              {GPU_PARAM_KEY(TILE_ID), target_tile_id}};
    return std::dynamic_pointer_cast<VAContext>(core.CreateContext(deviceName, contextParams)->GetHardwareContext());
}

/**
 * @brief This function is used to obtain remote blob object from VA surface handle
 * @param desc Tensor descriptor
 * @param ctx A remote context instance
 * @param surface A `VASurfaceID` to create remote blob from
 * @param plane An index of a plane inside `VASurfaceID` to create blob from
 * @return A remote blob wrapping `VASurfaceID`
 */
INFERENCE_ENGINE_1_0_DEPRECATED static inline VASurfaceBlob::Ptr make_shared_blob(const TensorDesc& desc,
                                                                                  RemoteContext::Ptr ctx,
                                                                                  VASurfaceID surface,
                                                                                  uint32_t plane = 0) {
    auto casted = ctx->as<VAContext>();
    if (nullptr == casted) {
        IE_THROW() << "Invalid remote context passed";
    }
    ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE)},
                       {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), surface},
                       {GPU_PARAM_KEY(VA_PLANE), plane}};
    return std::dynamic_pointer_cast<VASurfaceBlob>(casted->CreateBlob(desc, params));
}

}  // namespace gpu
}  // namespace InferenceEngine
