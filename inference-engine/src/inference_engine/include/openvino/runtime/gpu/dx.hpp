// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * shared Video Acceleration device contexts
 * and shared memory tensors which contain Video Acceleration surfaces
 *
 * @file openvino/runtime/gpu/dx.hpp
 */
#pragma once

#include <d3d11.h>

#include <memory>
#include <string>

#include "openvin/runtime/gpu/ocl.hpp"

namespace ov {
namespace runtime {
namespace gpu {

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which is shared with Direct3D 11 buffer.
 * The plugin object derived from this class can be obtained with D3DContext::create_tensor() call.
 * @note User can also obtain OpenCL buffer handle from this class.
 */
class D3DBufferTensor : public ClBufferTensor {
public:
    /**
     * @brief Checks that type defined runtime paramters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(
            tensor,
            {{GPU_PARAM_KEY(DEV_OBJECT_HANDLE), {}}, {GPU_PARAM_KEY(SHARED_MEM_TYPE), {GPU_PARAM_VALUE(DX_BUFFER)}}});
    }

    /**
     * @brief ID3D11Buffer conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Buffer interface
     */
    operator ID3D11Buffer*() {
        return static_cast<ID3D11Buffer*>(get_params().at(GPU_PARAM_KEY(DEV_OBJECT_HANDLE)).as<gpu_handle_param>());
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which is shared with Direct3D 11 2D texture.
 * The plugin object derived from this class can be obtained with D3DContext::create_tensor() call.
 * @note User can also obtain OpenCL 2D image handle from this class.
 */
class D3DSurface2DTensor : public ClImage2DTensor {
public:
    /**
     * @brief Checks that type defined runtime paramters are presented in remote object
     * @param remote_tensor remote tensor to check
     */
    static void type_check(const RemoteTensor& remote_tensor) {
        remote_type_check(remote_context.get_params(),
                          {{GPU_PARAM_KEY(DEV_OBJECT_HANDLE), {}},
                           {GPU_PARAM_KEY(VA_PLANE), {}},
                           {GPU_PARAM_KEY(SHARED_MEM_TYPE), {GPU_PARAM_VALUE(VA_SURFACE)}}});
    }

    /**
     * @brief ID3D11Texture2D conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Texture2D interface
     */
    operator ID3D11Texture2D*() {
        return static_cast<ID3D11Texture2D*>(get_params().at(GPU_PARAM_KEY(DEV_OBJECT_HANDLE)).as<gpu_handle_param>());
    }

    /**
     * @brief Returns plane ID of underlying video decoder surface, or 0 if no video surface was shared.
     * @return Plane ID
     */
    uint32_t plane() {
        return get_params().at(GPU_PARAM_KEY(VA_PLANE)).as<uint32_t>();
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with Direct3D 11 device.
 * The plugin object derived from this class can be obtained either with
 * ExecutableNetwork::get_context() or Core::create_context() calls.
 * @note User can also obtain OpenCL context handle from this class.
 */
class D3DContext : public ClContext {
    using RemoteContext::create_tensor;
    static constexpr const char* device_name = "GPU";

public:
    /**
     * @brief Checks that type defined runtime paramters are presented in remote object
     * @param remote_context remote context to check
     */
    static void type_check(const RemoteContext& remote_context) {
        remote_type_check(
            remote_context.get_params(),
            {{GPU_PARAM_KEY(VA_DEVICE), {}}, {GPU_PARAM_KEY(CONTEXT_TYPE), {GPU_PARAM_VALUE(VA_SHARED)}}});
    }

    /**
     * @brief ID3D11Device conversion operator for the D3DContext object.
     * @return Pointer to underlying ID3D11Device interface
     */
    operator ID3D11Device*() {
        return static_cast<ID3D11Device*>(get_params().at(GPU_PARAM_KEY(VA_DEVICE)).as<gpu_handle_param>());
    }

    /**
     * @brief Constructs D3DContext remote context object from ID3D11Device
     * @param core OpenVINO Runtime Core object instance
     * @param device A pointer to ID3D11Device to be used to create a remote context
     */
    D3DContext(Core& core, ID3D11Device* device) {
        // clang-format off
        ParamMap context_params = {
            {GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(VA_SHARED)},
            {GPU_PARAM_KEY(VA_DEVICE), static_cast<gpu_handle_param>(device)}
        };
        *this = core.create_context(device_name, context_params);
    }

    /**
     * @brief This function is used to obtain a NV12 tensor from NV12 DXGI video decoder output.
     * The resulting tensor contains two remote tensors for Y and UV planes of the surface.
     * @param height Height of Y plane
     * @param width Width of Y plane
     * @param nv12_surf A ID3D11Texture2D instance to create NV12 tensor from
     * @return A pair of remote tensors for each plane
     */
    std::pair<D3DSurface2DTensor, D3DSurface2DTensor> create_tensor_nv12(const size_t height, const size_t width, const ID3D11Texture2D* nv12_surf) {
        ParamMap tensor_params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE)},
                                  {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), static_cast<gpu_handle_param>(nv12_surf)},
                                  {GPU_PARAM_KEY(VA_PLANE), uint32_t(0)}};
        auto y_tensor = create_tensor(element::u8, {1, 1, height, width}, tensor_params);
        tensor_params[GPU_PARAM_KEY(MEM_HANDLE)] = static_cast<gpu_handle_param>(nv12_surf);
        tensor_params[GPU_PARAM_KEY(VA_PLANE)] = uint32_t(1);
        auto uv_tensor = create_tensor(element::u8, {1, 2, height / 2, width / 2}, tensor_params);
        return std::make_pair(y_tensor, uv_tensor);
    }

    /**
     * @brief This function is used to obtain remote tensor object from ID3D11Buffer
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param buffer A pointer to ID3D11Buffer instance to create remote tensor based on
     * @return A remote tensor instance
     */
    D3DBufferTensor create_tensor(const element::Type type, const Shape& shape, const ID3D11Buffer* buffer) {
        ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(DX_BUFFER)},
                           {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), static_cast<gpu_handle_param>(buffer)}};
        create_tensor(type, shape, params);
    }

    /**
     * @brief This function is used to obtain remote tensor object from ID3D11Texture2D
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param surface Pointer to ID3D11Texture2D interface of the objects that owns NV12 texture
     * @param plane ID of the plane to be shared (0 or 1)
     * @return D3DSurface2DTensor tensor
     * @note The underlying ID3D11Texture2D can also be a plane of output surface of DXGI video decoder
     */
    D3DSurface2DTensor create_tensor(const element::Type type,
                                     const Shape& shape,
                                     ID3D11Texture2D* surface,
                                     uint32_t plane = 0) {
        ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE)},
                           {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), static_cast<gpu_handle_param>(surface)},
                           {GPU_PARAM_KEY(VA_PLANE), plane}};
        return create_tensor(type, shape, params);
    }
};
}  // namespace gpu
}  // namespace runtime
}  // namespace ov
