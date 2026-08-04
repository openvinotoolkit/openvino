// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for properties of shared device contexts and shared device memory blobs for GPU plugin
 *        To use in constructors of Remote objects
 *
 * @file openvino/runtime/intel_gpu/remote_properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_gpu {

using gpu_handle_param = void*;

#ifdef __linux__
using os_handle_param = int;
#else
using os_handle_param = void*;
#endif

/**
 * @brief Enum to define the type of the shared context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
enum class ContextType {
    VULKAN = 0,  //!< Pure Vulkan context
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ContextType& context_type) {
    switch (context_type) {
    case ContextType::VULKAN:
        return os << "VULKAN";
    default:
        OPENVINO_THROW("Unsupported context type");
    }
}

inline std::istream& operator>>(std::istream& is, ContextType& context_type) {
    std::string str;
    is >> str;
    if (str == "VULKAN") {
        context_type = ContextType::VULKAN;
    } else {
        OPENVINO_THROW("Unsupported context type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief Shared device context type: can be either pure OpenCL (OCL)
 * or shared video decoder (VA_SHARED) context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<ContextType> context_type{"CONTEXT_TYPE"};

/**
 * @brief In case of multi-tile system,
 * this key identifies tile within given context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<int> tile_id{"TILE_ID"};

/**
 * @brief Enum to define the type of the shared memory buffer
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
enum class SharedMemType {
    USM_USER_BUFFER = 2,     //!< Shared USM pointer allocated by user
    USM_HOST_BUFFER = 3,     //!< Shared USM pointer type with host allocation type allocated by plugin
    USM_DEVICE_BUFFER = 4,   //!< Shared USM pointer type with device allocation type allocated by plugin
    BUFFER_FROM_HANDLE = 7,  //!< OS-level external memory handle (e.g. DX12 NT handle on Windows,
                             //!< DMA-BUF fd on Linux) imported by the plugin into a cl_mem
    CPU_VA = 8,              //!< Shared mmap-backed/aligned allocated host pointer mapped by plugin
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SharedMemType& share_mem_type) {
    switch (share_mem_type) {
    case SharedMemType::USM_USER_BUFFER:
        return os << "USM_USER_BUFFER";
    case SharedMemType::USM_HOST_BUFFER:
        return os << "USM_HOST_BUFFER";
    case SharedMemType::USM_DEVICE_BUFFER:
        return os << "USM_DEVICE_BUFFER";
    case SharedMemType::CPU_VA:
        return os << "CPU_VA";
    case SharedMemType::BUFFER_FROM_HANDLE:
        return os << "BUFFER_FROM_HANDLE";
    default:
        OPENVINO_THROW("Unsupported memory type");
    }
}

inline std::istream& operator>>(std::istream& is, SharedMemType& share_mem_type) {
    std::string str;
    is >> str;
    if (str == "USM_USER_BUFFER") {
        share_mem_type = SharedMemType::USM_USER_BUFFER;
    } else if (str == "USM_HOST_BUFFER") {
        share_mem_type = SharedMemType::USM_HOST_BUFFER;
    } else if (str == "USM_DEVICE_BUFFER") {
        share_mem_type = SharedMemType::USM_DEVICE_BUFFER;
    } else if (str == "CPU_VA") {
        share_mem_type = SharedMemType::CPU_VA;
    } else if (str == "BUFFER_FROM_HANDLE") {
        share_mem_type = SharedMemType::BUFFER_FROM_HANDLE;
    } else {
        OPENVINO_THROW("Unsupported memory type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief This key identifies type of internal shared memory
 * in a shared memory blob parameter map.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<SharedMemType> shared_mem_type{"SHARED_MEM_TYPE"};

/**
 * @brief This key identifies OpenCL memory handle
 * in a shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<gpu_handle_param> mem_handle{"MEM_HANDLE"};

/**
 * @brief This key identifies system memory handle (fd on Linux, NT handle on Windows)
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<os_handle_param> os_handle{"OS_HANDLE"};

/**
 * @brief This key identifies cpu pointer
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<void*> cpu_va{"CPU_VA"};

/**
 * @brief This key identifies size of allocated memory of cpu pointer
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<int64_t> cpu_va_size{"CPU_VA_SIZE"};

/**
 * @brief Platform OS memory handle for importing externally allocated memory into GPU plugin tensors.
 * On Linux this is a DMA-BUF file descriptor (int).
 * On Windows this is a DX12 shared NT HANDLE (void*).
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
struct SharedBufferHandle {
#ifdef __linux__
    using value_type = int;  ///< DMA-BUF file descriptor
#else
    using value_type = void*;  ///< DX12 shared NT HANDLE
#endif
    value_type value{};
};

/**
 * @brief Host (CPU) memory descriptor for wrapping mmap-backed or aligned host buffers
 * as GPU plugin tensors without copying.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
struct VirtualAddressMemory {
    explicit VirtualAddressMemory(void* ptr_, int64_t size_ = -1) : ptr(ptr_), size(size_) {}

    void* ptr = nullptr;
    int64_t size = -1;  ///< Buffer size in bytes; -1 means "derive from tensor shape"
};
}  // namespace intel_gpu
}  // namespace ov
