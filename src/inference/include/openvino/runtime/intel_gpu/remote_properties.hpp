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

#include <filesystem>

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
    OCL = 0,        //!< Pure OpenCL context
    VA_SHARED = 1,  //!< Context shared with a video decoding device
    ZE = 2,         //!< Pure Level Zero context
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ContextType& context_type) {
    switch (context_type) {
    case ContextType::OCL:
        return os << "OCL";
    case ContextType::VA_SHARED:
        return os << "VA_SHARED";
    case ContextType::ZE:
        return os << "ZE";
    default:
        OPENVINO_THROW("Unsupported context type");
    }
}

inline std::istream& operator>>(std::istream& is, ContextType& context_type) {
    std::string str;
    is >> str;
    if (str == "OCL") {
        context_type = ContextType::OCL;
    } else if (str == "VA_SHARED") {
        context_type = ContextType::VA_SHARED;
    } else if (str == "ZE") {
        context_type = ContextType::ZE;
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
 * @brief This key identifies OpenCL context handle
 * in a shared context or shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<gpu_handle_param> ocl_context{"OCL_CONTEXT"};

/**
 * @brief This key identifies ID of device in OpenCL context
 * if multiple devices are present in the context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<int> ocl_context_device_id{"OCL_CONTEXT_DEVICE_ID"};

/**
 * @brief In case of multi-tile system,
 * this key identifies tile within given context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<int> tile_id{"TILE_ID"};

/**
 * @brief This key identifies OpenCL queue handle in a shared context
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<gpu_handle_param> ocl_queue{"OCL_QUEUE"};

/**
 * @brief This key identifies video acceleration device/display handle
 * in a shared context or shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<gpu_handle_param> va_device{"VA_DEVICE"};

/**
 * @brief Enum to define the type of the shared memory buffer
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
enum class SharedMemType {
    OCL_BUFFER = 0,          //!< Shared OpenCL buffer blob
    OCL_IMAGE2D = 1,         //!< Shared OpenCL 2D image blob
    USM_USER_BUFFER = 2,     //!< Shared USM pointer allocated by user
    USM_HOST_BUFFER = 3,     //!< Shared USM pointer type with host allocation type allocated by plugin
    USM_DEVICE_BUFFER = 4,   //!< Shared USM pointer type with device allocation type allocated by plugin
    VA_SURFACE = 5,          //!< Shared video decoder surface or D3D 2D texture blob
    DX_BUFFER = 6,           //!< Shared D3D buffer blob
    BUFFER_FROM_HANDLE = 7,  //!< OS-level external memory handle (e.g. DX12 NT handle on Windows,
                             //!< DMA-BUF fd on Linux) imported by the plugin into a cl_mem
    CPU_VA = 8,              //!< Shared mmap-backed/aligned allocated host pointer mapped by plugin
    MMAPED_FILE = 9,         //!< Memory-mapped file buffer read and wrapped by the plugin
};

/**
 * @brief Enum to define memory type for pointer-based tensor sharing API.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
enum class MemType {
    SHARED_BUF = 0,  //!< Shared OpenCL buffer handle passed as void* or int
    CPU_VA = 1,      //!< CPU Virtual Address buffer
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SharedMemType& share_mem_type) {
    switch (share_mem_type) {
    case SharedMemType::OCL_BUFFER:
        return os << "OCL_BUFFER";
    case SharedMemType::OCL_IMAGE2D:
        return os << "OCL_IMAGE2D";
    case SharedMemType::USM_USER_BUFFER:
        return os << "USM_USER_BUFFER";
    case SharedMemType::USM_HOST_BUFFER:
        return os << "USM_HOST_BUFFER";
    case SharedMemType::USM_DEVICE_BUFFER:
        return os << "USM_DEVICE_BUFFER";
    case SharedMemType::CPU_VA:
        return os << "CPU_VA";
    case SharedMemType::VA_SURFACE:
        return os << "VA_SURFACE";
    case SharedMemType::DX_BUFFER:
        return os << "DX_BUFFER";
    case SharedMemType::BUFFER_FROM_HANDLE:
        return os << "BUFFER_FROM_HANDLE";
    case SharedMemType::MMAPED_FILE:
        return os << "MMAPED_FILE";
    default:
        OPENVINO_THROW("Unsupported memory type");
    }
}

inline std::istream& operator>>(std::istream& is, SharedMemType& share_mem_type) {
    std::string str;
    is >> str;
    if (str == "OCL_BUFFER") {
        share_mem_type = SharedMemType::OCL_BUFFER;
    } else if (str == "OCL_IMAGE2D") {
        share_mem_type = SharedMemType::OCL_IMAGE2D;
    } else if (str == "USM_USER_BUFFER") {
        share_mem_type = SharedMemType::USM_USER_BUFFER;
    } else if (str == "USM_HOST_BUFFER") {
        share_mem_type = SharedMemType::USM_HOST_BUFFER;
    } else if (str == "USM_DEVICE_BUFFER") {
        share_mem_type = SharedMemType::USM_DEVICE_BUFFER;
    } else if (str == "CPU_VA") {
        share_mem_type = SharedMemType::CPU_VA;
    } else if (str == "VA_SURFACE") {
        share_mem_type = SharedMemType::VA_SURFACE;
    } else if (str == "DX_BUFFER") {
        share_mem_type = SharedMemType::DX_BUFFER;
    } else if (str == "BUFFER_FROM_HANDLE") {
        share_mem_type = SharedMemType::BUFFER_FROM_HANDLE;
    } else if (str == "MMAPED_FILE") {
        share_mem_type = SharedMemType::MMAPED_FILE;
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
 * @brief Enum to define access mode of memory shared with the plugin (host virtual-address buffers,
 * memory-mapped files, etc.)
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
enum class AccessMode {
    READ = 0,       //!< Tensor data is only read
    READ_WRITE = 1  //!< Tensor data is also written back; requires writable memory (or a writable file)
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const AccessMode& access_mode) {
    switch (access_mode) {
    case AccessMode::READ:
        return os << "READ";
    case AccessMode::READ_WRITE:
        return os << "READ_WRITE";
    default:
        OPENVINO_THROW("Unsupported access mode");
    }
}

inline std::istream& operator>>(std::istream& is, AccessMode& access_mode) {
    std::string str;
    is >> str;
    if (str == "READ") {
        access_mode = AccessMode::READ;
    } else if (str == "READ_WRITE") {
        access_mode = AccessMode::READ_WRITE;
    } else {
        OPENVINO_THROW("Unsupported access mode: ", str);
    }
    return is;
}
/** @endcond */

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
 * @brief This key identifies access mode (read or read-write) of the memory pointed to by cpu_va
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<AccessMode> cpu_va_access{"CPU_VA_ACCESS"};

/**
 * @brief This key identifies video decoder surface handle
 * in a shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
#ifdef _WIN32
static constexpr Property<gpu_handle_param> dev_object_handle{"DEV_OBJECT_HANDLE"};
#else
static constexpr Property<uint32_t> dev_object_handle{"DEV_OBJECT_HANDLE"};
#endif

/**
 * @brief This key identifies video decoder surface plane
 * in a shared memory blob parameter map
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<uint32_t> va_plane{"VA_PLANE"};

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
    explicit VirtualAddressMemory(void* ptr_, int64_t size_ = -1, AccessMode access_ = AccessMode::READ_WRITE)
        : ptr(ptr_),
          size(size_),
          access(access_) {}

    void* ptr = nullptr;
    int64_t size = -1;                           ///< Buffer size in bytes; -1 means "derive from tensor shape"
    AccessMode access = AccessMode::READ_WRITE;  ///< Whether the plugin may only read this memory or also write to it
};

/**
 * @brief File descriptor for wrapping tensor data memory-mapped from a file as a GPU plugin tensor.
 * The plugin memory-maps the file and keeps the mapping alive for the whole tensor lifetime,
 * so the file must not be modified until the returned tensor is destroyed.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
struct FileDescriptor {  // need to be merged with ov::intel_npu::FileDescriptor in future
    explicit FileDescriptor(const std::filesystem::path& file_path,
                            std::size_t offset_in_bytes = 0,
                            AccessMode file_access = AccessMode::READ)
        : path(file_path),
          offset(offset_in_bytes),
          access(file_access) {
        OPENVINO_ASSERT(!file_path.empty(), "[GPU] Provided file path is empty.");
    }

    std::filesystem::path path;            ///< File path
    std::size_t offset = 0;                ///< Offset in bytes to read from the file
    AccessMode access = AccessMode::READ;  ///< Access mode of the mapping
};

/**
 * @brief This key identifies the file descriptor
 * in a memory-mapped tensor parameter map.
 * @ingroup ov_runtime_ocl_gpu_cpp_api
 */
static constexpr Property<FileDescriptor> file_descriptor{"FILE_DESCRIPTOR"};
}  // namespace intel_gpu
}  // namespace ov
