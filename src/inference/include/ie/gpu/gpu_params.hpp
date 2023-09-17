// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header for properties
 * of shared device contexts and shared device memory blobs
 * for clDNN plugin
 *
 * @file gpu_params.hpp
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

#include <string>

#include "ie_api.h"

namespace InferenceEngine {
/**
 * @brief Shortcut for defining a handle parameter
 */
using gpu_handle_param = void*;

namespace GPUContextParams {
/**
 * @def GPU_PARAM_KEY(name)
 * @brief Shortcut for defining configuration keys
 */
#define GPU_PARAM_KEY(name) ::InferenceEngine::GPUContextParams::PARAM_##name
/**
 * @def GPU_PARAM_VALUE(name)
 * @brief Shortcut for defining configuration values
 */
#define GPU_PARAM_VALUE(name) ::InferenceEngine::GPUContextParams::name

/**
 * @def DECLARE_GPU_PARAM_VALUE(name)
 * @brief Shortcut for defining possible values for object parameter keys
 */
#define DECLARE_GPU_PARAM_VALUE(name) static constexpr auto name = #name

/**
 * @def DECLARE_GPU_PARAM_KEY(name, ...)
 * @brief Shortcut for defining object parameter keys
 */
#define DECLARE_GPU_PARAM_KEY(name, ...) static constexpr auto PARAM_##name = #name
/**
 * @brief Shared device context type: can be either pure OpenCL (OCL)
 * or shared video decoder (VA_SHARED) context
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(CONTEXT_TYPE, std::string);
/**
 * @brief Pure OpenCL device context
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(OCL);
/**
 * @brief Shared context (video decoder or D3D)
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(VA_SHARED);

/**
 * @brief This key identifies OpenCL context handle
 * in a shared context or shared memory blob parameter map
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(OCL_CONTEXT, gpu_handle_param);

/**
 * @brief This key identifies ID of device in OpenCL context
 * if multiple devices are present in the context
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(OCL_CONTEXT_DEVICE_ID, int);

/**
 * @brief In case of multi-tile system,
 * this key identifies tile within given context
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(TILE_ID, int);

/**
 * @brief This key identifies OpenCL queue handle in a shared context
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(OCL_QUEUE, gpu_handle_param);

/**
 * @brief This key identifies video acceleration device/display handle
 * in a shared context or shared memory blob parameter map
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(VA_DEVICE, gpu_handle_param);

/**
 * @brief This key identifies type of internal shared memory
 * in a shared memory blob parameter map.
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(SHARED_MEM_TYPE, std::string);
/**
 * @brief Shared OpenCL buffer blob
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(OCL_BUFFER);
/**
 * @brief Shared OpenCL 2D image blob
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(OCL_IMAGE2D);
/**
 * @brief Shared USM pointer allocated by user
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(USM_USER_BUFFER);
/**
 * @brief Shared USM pointer type with host allocation type allocated by plugin
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(USM_HOST_BUFFER);
/**
 * @brief Shared USM pointer type with device allocation type allocated by plugin
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(USM_DEVICE_BUFFER);
/**
 * @brief Shared video decoder surface or D3D 2D texture blob
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(VA_SURFACE);

/**
 * @brief Shared D3D buffer blob
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_VALUE(DX_BUFFER);

/**
 * @brief This key identifies OpenCL memory handle
 * in a shared memory blob parameter map
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(MEM_HANDLE, gpu_handle_param);

/**
 * @brief This key identifies video decoder surface handle
 * in a shared memory blob parameter map
 */
#ifdef _WIN32
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(DEV_OBJECT_HANDLE, gpu_handle_param);
#else
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(DEV_OBJECT_HANDLE, uint32_t);
#endif

/**
 * @brief This key identifies video decoder surface plane
 * in a shared memory blob parameter map
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_GPU_PARAM_KEY(VA_PLANE, uint32_t);

}  // namespace GPUContextParams
}  // namespace InferenceEngine
