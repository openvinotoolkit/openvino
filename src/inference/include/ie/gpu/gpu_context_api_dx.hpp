// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * shared Video Acceleration device contexts
 * and shared memory blobs which contain Video Acceleration surfaces
 *
 * @file gpu_context_api_dx.hpp
 */
#pragma once

#include <d3d11.h>

#include <memory>
#include <string>

#include "gpu/gpu_context_api_ocl.hpp"

namespace InferenceEngine {

namespace gpu {
/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with Direct3D 11 device.
 * The plugin object derived from this class can be obtained either with
 * GetContext() method of Executable network or using CreateContext() Core call.
 * @note User can also obtain OpenCL context handle from this class.
 */
class D3DContext : public ClContext {
public:
    /**
     * @brief A smart pointer to the D3DContext object
     */
    using Ptr = std::shared_ptr<D3DContext>;

    /**
     * @brief ID3D11Device conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Device interface
     */
    operator ID3D11Device*() {
        return _ObjFromParams<ID3D11Device*, gpu_handle_param>(getParams(),
                                                               GPU_PARAM_KEY(VA_DEVICE),
                                                               GPU_PARAM_KEY(CONTEXT_TYPE),
                                                               GPU_PARAM_VALUE(VA_SHARED));
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote blob
 * which is shared with Direct3D 11 buffer.
 * The plugin object derived from this class can be obtained with CreateBlob() call.
 * @note User can also obtain OpenCL buffer handle from this class.
 */
class D3DBufferBlob : public ClBufferBlob {
public:
    /**
     * @brief A smart pointer to the D3DBufferBlob object
     */
    using Ptr = std::shared_ptr<D3DBufferBlob>;

    /**
     * @brief Creates a D3DBufferBlob object with the specified dimensions and layout.
     * @param tensorDesc Tensor description
     */
    explicit D3DBufferBlob(const TensorDesc& tensorDesc) : ClBufferBlob(tensorDesc) {}

    /**
     * @brief ID3D11Buffer conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Buffer interface
     */
    operator ID3D11Buffer*() {
        return _ObjFromParams<ID3D11Buffer*, gpu_handle_param>(getParams(),
                                                               GPU_PARAM_KEY(DEV_OBJECT_HANDLE),
                                                               GPU_PARAM_KEY(SHARED_MEM_TYPE),
                                                               GPU_PARAM_VALUE(DX_BUFFER));
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote blob
 * which is shared with Direct3D 11 2D texture.
 * The plugin object derived from this class can be obtained with CreateBlob() call.
 * @note User can also obtain OpenCL 2D image handle from this class.
 */
class D3DSurface2DBlob : public ClImage2DBlob {
public:
    /**
     * @brief A smart pointer to the D3DSurface2DBlob object
     */
    using Ptr = std::shared_ptr<D3DSurface2DBlob>;

    /**
     * @brief Creates a D3DSurface2DBlob object with the specified dimensions and layout.
     * @param tensorDesc Tensor description
     */
    explicit D3DSurface2DBlob(const TensorDesc& tensorDesc) : ClImage2DBlob(tensorDesc) {}

    /**
     * @brief ID3D11Texture2D conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Texture2D interface
     */
    operator ID3D11Texture2D*() {
        return _ObjFromParams<ID3D11Texture2D*, gpu_handle_param>(getParams(),
                                                                  GPU_PARAM_KEY(DEV_OBJECT_HANDLE),
                                                                  GPU_PARAM_KEY(SHARED_MEM_TYPE),
                                                                  GPU_PARAM_VALUE(VA_SURFACE));
    }

    /**
     * @brief Returns plane ID of underlying video decoder surface, or 0 if no video surface was shared.
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
 * @brief This function is used to obtain a NV12 compound blob object from NV12 DXGI video decoder output.
 * The resulting compound contains two remote blobs for Y and UV planes of the surface.
 * @param height Height of Y plane
 * @param width Widht of Y plane
 * @param ctx A pointer to remote context
 * @param nv12_surf A ID3D11Texture2D instance to create NV12 blob from
 * @return NV12 remote blob
 */
static inline Blob::Ptr make_shared_blob_nv12(size_t height,
                                              size_t width,
                                              RemoteContext::Ptr ctx,
                                              ID3D11Texture2D* nv12_surf) {
    // despite of layout, blob dimensions always follow in N,C,H,W order
    TensorDesc desc(Precision::U8, {1, 1, height, width}, Layout::NHWC);

    ParamMap blobParams = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE)},
                           {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), static_cast<gpu_handle_param>(nv12_surf)},
                           {GPU_PARAM_KEY(VA_PLANE), uint32_t(0)}};
    Blob::Ptr y_blob = std::dynamic_pointer_cast<Blob>(ctx->CreateBlob(desc, blobParams));

    TensorDesc uvdesc(Precision::U8, {1, 2, height / 2, width / 2}, Layout::NHWC);
    blobParams[GPU_PARAM_KEY(MEM_HANDLE)] = static_cast<gpu_handle_param>(nv12_surf);
    blobParams[GPU_PARAM_KEY(VA_PLANE)] = uint32_t(1);
    Blob::Ptr uv_blob = std::dynamic_pointer_cast<Blob>(ctx->CreateBlob(uvdesc, blobParams));

    return InferenceEngine::make_shared_blob<NV12Blob>(y_blob, uv_blob);
}

/**
 * @brief This function is used to obtain remote context object from ID3D11Device
 * @param core Inference Engine Core object instance
 * @param deviceName A name of to create a remote context for
 * @param device A pointer to ID3D11Device to be used to create a remote context
 * @param target_tile_id Desired tile id within given context for multi-tile system. Default value (-1) means that root
 * device should be used
 * @return A shared remote context instance
 */
static inline D3DContext::Ptr make_shared_context(Core& core,
                                                  std::string deviceName,
                                                  ID3D11Device* device,
                                                  int target_tile_id = -1) {
    // clang-format off
    ParamMap contextParams = {
        {GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(VA_SHARED)},
        {GPU_PARAM_KEY(VA_DEVICE), static_cast<gpu_handle_param>(device)},
        {GPU_PARAM_KEY(TILE_ID), target_tile_id}
    };
    // clang-format on
    return std::dynamic_pointer_cast<D3DContext>(core.CreateContext(deviceName, contextParams));
}

/**
 * @brief This function is used to obtain remote blob object from ID3D11Buffer
 * @param desc A tensor description which describes blob configuration
 * @param ctx A shared pointer to a remote context
 * @param buffer A pointer to ID3D11Buffer instance to create remote blob based on
 * @return A remote blob instance
 */
static inline Blob::Ptr make_shared_blob(const TensorDesc& desc, RemoteContext::Ptr ctx, ID3D11Buffer* buffer) {
    auto casted = std::dynamic_pointer_cast<D3DContext>(ctx);
    if (nullptr == casted) {
        IE_THROW() << "Invalid remote context passed";
    }

    ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(DX_BUFFER)},
                       {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), static_cast<gpu_handle_param>(buffer)}};
    return std::dynamic_pointer_cast<D3DBufferBlob>(casted->CreateBlob(desc, params));
}

/**
 * @brief This function is used to obtain remote blob object from ID3D11Texture2D
 * @param desc Tensor description
 * @param ctx the RemoteContext object whuch owns context for the blob to be created
 * @param surface Pointer to ID3D11Texture2D interface of the objects that owns NV12 texture
 * @param plane ID of the plane to be shared (0 or 1)
 * @return Smart pointer to created RemoteBlob object cast to base class
 * @note The underlying ID3D11Texture2D can also be a plane of output surface of DXGI video decoder
 */
static inline Blob::Ptr make_shared_blob(const TensorDesc& desc,
                                         RemoteContext::Ptr ctx,
                                         ID3D11Texture2D* surface,
                                         uint32_t plane = 0) {
    auto casted = std::dynamic_pointer_cast<D3DContext>(ctx);
    if (nullptr == casted) {
        IE_THROW() << "Invalid remote context passed";
    }
    ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE)},
                       {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), static_cast<gpu_handle_param>(surface)},
                       {GPU_PARAM_KEY(VA_PLANE), plane}};
    return std::dynamic_pointer_cast<D3DSurface2DBlob>(casted->CreateBlob(desc, params));
}

}  // namespace gpu
}  // namespace InferenceEngine
