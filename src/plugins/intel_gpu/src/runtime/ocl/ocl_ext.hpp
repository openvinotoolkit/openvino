// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// \file This file wraps cl2.hpp and introduces wrapper classes for Intel sharing extensions.
///

#pragma once

#include <array>

#ifdef OV_GPU_USE_OPENCL_HPP
#include <CL/opencl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#ifndef CL_HPP_PARAM_NAME_CL_INTEL_UNIFIED_SHARED_MEMORY_
#define OPENVINO_CLHPP_HEADERS_ARE_OLDER_THAN_V2024_10_24
#endif

#include <CL/cl_ext.h>

#ifdef _WIN32
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# include <CL/cl_d3d11.h>
typedef cl_d3d11_device_source_khr cl_device_source_intel;
typedef cl_d3d11_device_set_khr    cl_device_set_intel;
#else
# include <CL/cl_va_api_media_sharing_intel.h>
typedef cl_va_api_device_source_intel cl_device_source_intel;
typedef cl_va_api_device_set_intel    cl_device_set_intel;
#endif

#include <sstream>

/********************************************
* cl_intel_required_subgroup_size extension *
*********************************************/

#if !defined(cl_intel_required_subgroup_size)
#define cl_intel_required_subgroup_size 1

// cl_intel_required_subgroup_size
#define CL_DEVICE_SUB_GROUP_SIZES_INTEL           0x4108

#endif // cl_intel_required_subgroup_size

#ifdef OPENVINO_CLHPP_HEADERS_ARE_OLDER_THAN_V2024_10_24

namespace cl {
namespace detail {
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SUB_GROUP_SIZES_INTEL, cl::vector<size_type>)
}  // namespace detail
}  // namespace cl

#endif // OPENVINO_CLHPP_HEADERS_ARE_OLDER_THAN_V2024_10_24

/***************************************************************
* cl_intel_command_queue_families
***************************************************************/

#if !defined(cl_intel_command_queue_families)
#define cl_intel_command_queue_families 1

typedef cl_bitfield         cl_command_queue_capabilities_intel;

#define CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL                 64

typedef struct _cl_queue_family_properties_intel {
    cl_command_queue_properties properties;
    cl_command_queue_capabilities_intel capabilities;
    cl_uint count;
    char name[CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL];
} cl_queue_family_properties_intel;

/* cl_device_info */
#define CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL             0x418B

/* cl_queue_properties */
#define CL_QUEUE_FAMILY_INTEL                               0x418C
#define CL_QUEUE_INDEX_INTEL                                0x418D

/* cl_command_queue_capabilities_intel */
#define CL_QUEUE_DEFAULT_CAPABILITIES_INTEL                 0

#endif  // cl_intel_command_queue_families

/*******************************************
* cl_intel_unified_shared_memory extension *
********************************************/

#if !defined(cl_intel_unified_shared_memory)
#define cl_intel_unified_shared_memory 1

/* cl_mem_alloc_info_intel */
#define CL_MEM_ALLOC_TYPE_INTEL         0x419A
#define CL_MEM_ALLOC_SIZE_INTEL         0x419C

/* cl_unified_shared_memory_type_intel */
#define CL_MEM_TYPE_UNKNOWN_INTEL       0x4196
#define CL_MEM_TYPE_HOST_INTEL          0x4197
#define CL_MEM_TYPE_DEVICE_INTEL        0x4198
#define CL_MEM_TYPE_SHARED_INTEL        0x4199

/* cl_device_info */
#define CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL                   0x4190
#define CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL                 0x4191
#define CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL   0x4192

/* cl_device_unified_shared_memory_capabilities_intel - bitfield */
#define CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL                   (1 << 0)

typedef cl_ulong            cl_properties;
typedef cl_properties cl_mem_properties_intel;
typedef cl_uint cl_mem_info_intel;
typedef cl_uint cl_unified_shared_memory_type_intel;
typedef cl_bitfield cl_device_unified_shared_memory_capabilities_intel;

typedef cl_int (CL_API_CALL *
clMemFreeINTEL_fn)(
            cl_context context,
            void* ptr);

typedef cl_int (CL_API_CALL *
clSetKernelArgMemPointerINTEL_fn)(
            cl_kernel kernel,
            cl_uint arg_index,
            const void* arg_value);

typedef cl_int (CL_API_CALL *
clEnqueueMemcpyINTEL_fn)(
            cl_command_queue command_queue,
            cl_bool blocking,
            void* dst_ptr,
            const void* src_ptr,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef void* (CL_API_CALL *
clHostMemAllocINTEL_fn)(
            cl_context context,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef void* (CL_API_CALL *
clSharedMemAllocINTEL_fn)(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef void* (CL_API_CALL *
clDeviceMemAllocINTEL_fn)(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef cl_int (CL_API_CALL *
clGetMemAllocInfoINTEL_fn)(
            cl_context context,
            const void* ptr,
            cl_mem_info_intel param_name,
            size_t param_value_size,
            void* param_value,
            size_t* param_value_size_ret);

typedef cl_int (CL_API_CALL *
clEnqueueMemsetINTEL_fn)(   /* Deprecated */
            cl_command_queue command_queue,
            void* dst_ptr,
            cl_int value,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef cl_int (CL_API_CALL *
clEnqueueMemFillINTEL_fn)(
            cl_command_queue command_queue,
            void* dst_ptr,
            const void* pattern,
            size_t pattern_size,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

#endif // cl_intel_unified_shared_memory

/********************************
* cl_intel_planar_yuv extension *
*********************************/

#if !defined(CL_NV12_INTEL)
#define CL_NV12_INTEL                                       0x410E
#endif // CL_NV12_INTEL

#if !defined(CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL)
#define CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL              (1 << 25)
#endif // CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL

/*********************************
* cl_khr_device_uuid extension
*********************************/

#if !defined(cl_khr_device_uuid)
#define cl_khr_device_uuid 1

#define CL_UUID_SIZE_KHR 16
#define CL_LUID_SIZE_KHR 8

#define CL_DEVICE_UUID_KHR          0x106A
#define CL_DEVICE_LUID_KHR          0x106D

#endif // cl_khr_device_uuid

// some versions of CL/opencl.hpp don't define C++ wrapper for CL_DEVICE_UUID_KHR
// we are checking it in cmake and defined macro OV_GPU_OPENCL_HPP_HAS_UUID if it is defined
#ifndef OV_GPU_OPENCL_HPP_HAS_UUID

// for C++ wrappers
using uuid_array = std::array<cl_uchar, CL_UUID_SIZE_KHR>;
using luid_array = std::array<cl_uchar, CL_LUID_SIZE_KHR>;

namespace cl {
namespace detail {
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_UUID_KHR, uuid_array)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LUID_KHR, luid_array)
}  // namespace detail
}  // namespace cl

#endif // OV_GPU_OPENCL_HPP_HAS_UUID

/***************************************************************
* cl_intel_device_attribute_query
***************************************************************/

#if !defined(cl_intel_device_attribute_query)
#define cl_intel_device_attribute_query 1

typedef cl_bitfield         cl_device_feature_capabilities_intel;

/* cl_device_feature_capabilities_intel */
#define CL_DEVICE_FEATURE_FLAG_DP4A_INTEL                   (1 << 0)
#define CL_DEVICE_FEATURE_FLAG_DPAS_INTEL                   (1 << 1)

/* cl_device_info */
#define CL_DEVICE_IP_VERSION_INTEL                          0x4250
#define CL_DEVICE_ID_INTEL                                  0x4251
#define CL_DEVICE_NUM_SLICES_INTEL                          0x4252
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL            0x4253
#define CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL               0x4254
#define CL_DEVICE_NUM_THREADS_PER_EU_INTEL                  0x4255
#define CL_DEVICE_FEATURE_CAPABILITIES_INTEL                0x4256

#endif // cl_intel_device_attribute_query

#ifndef CL_HPP_PARAM_NAME_CL_INTEL_COMMAND_QUEUE_FAMILIES_
#define CL_HPP_PARAM_NAME_CL_INTEL_COMMAND_QUEUE_FAMILIES_(F) \
    F(cl_device_info, CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL, cl::vector<cl_queue_family_properties_intel>) \
    \
    F(cl_command_queue_info, CL_QUEUE_FAMILY_INTEL, cl_uint) \
    F(cl_command_queue_info, CL_QUEUE_INDEX_INTEL, cl_uint)
#endif // CL_HPP_PARAM_NAME_CL_INTEL_COMMAND_QUEUE_FAMILIES_

#ifdef OPENVINO_CLHPP_HEADERS_ARE_OLDER_THAN_V2024_10_24

namespace cl {
namespace detail {
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_IP_VERSION_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_ID_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_SLICES_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_THREADS_PER_EU_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_FEATURE_CAPABILITIES_INTEL, cl_device_feature_capabilities_intel)
CL_HPP_PARAM_NAME_CL_INTEL_COMMAND_QUEUE_FAMILIES_(CL_HPP_DECLARE_PARAM_TRAITS_)
}  // namespace detail
}  // namespace cl

#endif // OPENVINO_CLHPP_HEADERS_ARE_OLDER_THAN_V2024_10_24

#include <memory>

namespace {
template <typename T>
T load_entrypoint(const cl_platform_id platform, const std::string name) {
#if defined(__GNUC__) && __GNUC__ < 5
// OCL spec says:
// "The function clGetExtensionFunctionAddressForPlatform returns the address of the extension function named by funcname for a given platform.
//  The pointer returned should be cast to a function pointer type matching the extension function's definition defined in the appropriate extension
//  specification and header file."
// So the pointer-to-object to pointer-to-function cast below is supposed to be valid, thus we suppress warning from old GCC versions.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    T p = reinterpret_cast<T>(clGetExtensionFunctionAddressForPlatform(platform, name.c_str()));
#if defined(__GNUC__) && __GNUC__ < 5
#pragma GCC diagnostic pop
#endif
    if (!p) {
        throw std::runtime_error("clGetExtensionFunctionAddressForPlatform(" + name + ") returned NULL.");
    }
    return p;
}

template <typename T>
T load_entrypoint(const cl_device_id device, const std::string name) {
    cl_platform_id platform;
    cl_int error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_DEVICE_PLATFORM: " + std::to_string(error));
    }
    return load_entrypoint<T>(platform, name);
}

// loader functions created for single device contexts
// ToDo Extend it for multi device case.
template <typename T>
T load_entrypoint(const cl_context context, const std::string name) {
    size_t size = 0;
    cl_int error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &size);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_CONTEXT_DEVICES size: " + std::to_string(error));
    }

    std::vector<cl_device_id> devices(size / sizeof(cl_device_id));

    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices.data(), nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_CONTEXT_DEVICES: " + std::to_string(error));
    }

    return load_entrypoint<T>(devices.front(), name);
}

template <typename T>
T try_load_entrypoint(const cl_context context, const std::string name) {
    try {
        return load_entrypoint<T>(context, name);
    } catch (...) {
        return nullptr;
    }
}

template <typename T>
T try_load_entrypoint(const cl_platform_id platform, const std::string name) {
    try {
        return load_entrypoint<T>(platform, name);
    } catch (...) {
        return nullptr;
    }
}

template <typename T>
T load_entrypoint(const cl_kernel kernel, const std::string name) {
    cl_context context;
    cl_int error = clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(context),
        &context, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_KERNEL_CONTEXT: " +
            std::to_string(error));
    }
    return load_entrypoint<T>(context, name);
}

template <typename T>
T load_entrypoint(const cl_command_queue queue, const std::string name) {
    cl_context context;
    cl_int error = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context),
        &context, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_QUEUE_CONTEXT: " +
            std::to_string(error));
    }
    return load_entrypoint<T>(context, name);
}
}  // namespace


namespace cl {

typedef CL_API_ENTRY cl_int(CL_API_CALL *PFN_clEnqueueAcquireMediaSurfacesINTEL)(
    cl_command_queue /* command_queue */,
    cl_uint /* num_objects */,
    const cl_mem* /* mem_objects */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */,
    cl_event* /* event */);

typedef CL_API_ENTRY cl_int(CL_API_CALL *PFN_clEnqueueReleaseMediaSurfacesINTEL)(
    cl_command_queue /* command_queue */,
    cl_uint /* num_objects */,
    const cl_mem* /* mem_objects */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */,
    cl_event* /* event */);

typedef CL_API_ENTRY cl_mem(CL_API_CALL * PFN_clCreateFromMediaSurfaceINTEL)(
    cl_context /* context */,
    cl_mem_flags /* flags */,
    void* /* surface */,
    cl_uint /* plane */,
    cl_int* /* errcode_ret */);


#ifdef WIN32
    typedef CL_API_ENTRY cl_mem(CL_API_CALL * PFN_clCreateFromD3D11Buffer)(
        cl_context context,
        cl_mem_flags flags,
        void* resource, cl_int* errcode_ret);
#endif

class SharedSurfLock {
    cl_command_queue m_queue;
    std::vector<cl_mem> m_surfaces;
    cl_int* m_errPtr;

public:
    static PFN_clEnqueueAcquireMediaSurfacesINTEL pfn_acquire;
    static PFN_clEnqueueReleaseMediaSurfacesINTEL pfn_release;

    static void Init(cl_platform_id platform) {
#ifdef WIN32
        const char* fnameAcq = "clEnqueueAcquireD3D11ObjectsKHR";
        const char* fnameRel = "clEnqueueReleaseD3D11ObjectsKHR";
#else
        const char* fnameAcq = "clEnqueueAcquireVA_APIMediaSurfacesINTEL";
        const char* fnameRel = "clEnqueueReleaseVA_APIMediaSurfacesINTEL";
#endif
        if (!pfn_acquire) {
            pfn_acquire = try_load_entrypoint<PFN_clEnqueueAcquireMediaSurfacesINTEL>(platform, fnameAcq);
        }
        if (!pfn_release) {
            pfn_release = try_load_entrypoint<PFN_clEnqueueReleaseMediaSurfacesINTEL>(platform, fnameRel);
        }
    }

    SharedSurfLock(cl_command_queue queue,
        std::vector<cl_mem>& surfaces,
        cl_int * err = NULL)
        : m_queue(queue), m_surfaces(surfaces), m_errPtr(err) {
        if (pfn_acquire != NULL && m_surfaces.size()) {
            cl_int error = pfn_acquire(m_queue,
                static_cast<cl_uint>(m_surfaces.size()),
                m_surfaces.data(),
                0, NULL, NULL);

            if (error != CL_SUCCESS && m_errPtr != NULL) {
                *m_errPtr = error;
            }
        }
    }

    ~SharedSurfLock() {
        if (pfn_release != NULL && m_surfaces.size()) {
            cl_int error = pfn_release(m_queue,
                static_cast<cl_uint>(m_surfaces.size()),
                m_surfaces.data(),
                0, NULL, NULL);
            if (error != CL_SUCCESS && m_errPtr != NULL) {
                *m_errPtr = error;
            }
        }
    }
};

class ImageVA : public Image2D {
public:
    static PFN_clCreateFromMediaSurfaceINTEL pfn_clCreateFromMediaSurfaceINTEL;

    static void Init(cl_platform_id platform) {
#ifdef WIN32
        const char* fname = "clCreateFromD3D11Texture2DKHR";
#else
        const char* fname = "clCreateFromVA_APIMediaSurfaceINTEL";
#endif
        if (!pfn_clCreateFromMediaSurfaceINTEL) {
            pfn_clCreateFromMediaSurfaceINTEL = try_load_entrypoint<PFN_clCreateFromMediaSurfaceINTEL>((cl_platform_id)platform, fname);
        }
    }

    /*! \brief Constructs a ImageVA, in a specified context, from a
    *         given vaSurfaceID.
    *
    *  Wraps clCreateFromMediaSurfaceINTEL().
    */
    ImageVA(
        const Context& context,
        cl_mem_flags flags,
#ifdef WIN32
        void* surface,
#else
        uint32_t surface,
#endif
        uint32_t plane,
        cl_int * err = NULL) {
        cl_int error;
        object_ = pfn_clCreateFromMediaSurfaceINTEL(
            context(),
            flags,
#ifdef WIN32
            surface,
#else
            reinterpret_cast<void*>(&surface),
#endif
            plane,
            &error);

        detail::errHandler(error);
        if (err != NULL) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to NULL.
    ImageVA() : Image2D() { }

    /*! \brief Constructor from cl_mem - takes ownership.
    *
    * \param retainObject will cause the constructor to retain its cl object.
    *                     Defaults to false to maintain compatibility with
    *                     earlier versions.
    *  See Memory for further details.
    */
    explicit ImageVA(const cl_mem& image, bool retainObject = false) :
        Image2D(image, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
    *
    *  See Memory for further details.
    */
    ImageVA& operator = (const cl_mem& rhs) {
        Image2D::operator=(rhs);
        return *this;
    }

    /*! \brief Copy constructor to forward copy to the superclass correctly.
    * Required for MSVC.
    */
    ImageVA(const ImageVA& img) :
        Image2D(img) {}

    /*! \brief Copy assignment to forward copy to the superclass correctly.
    * Required for MSVC.
    */
    ImageVA& operator = (const ImageVA &img) {
        Image2D::operator=(img);
        return *this;
    }

    /*! \brief Move constructor to forward move to the superclass correctly.
    * Required for MSVC.
    */
    ImageVA(ImageVA&& buf) noexcept : Image2D(std::move(buf)) {}

    /*! \brief Move assignment to forward move to the superclass correctly.
    * Required for MSVC.
    */
    ImageVA& operator = (ImageVA &&buf) {
        Image2D::operator=(std::move(buf));
        return *this;
    }
};
#ifdef WIN32
class BufferDX : public Buffer {
public:
    static PFN_clCreateFromD3D11Buffer pfn_clCreateFromD3D11Buffer;

    static void Init(cl_platform_id platform) {
        const char* fname = "clCreateFromD3D11BufferKHR";

        if (!pfn_clCreateFromD3D11Buffer) {
            pfn_clCreateFromD3D11Buffer = try_load_entrypoint<PFN_clCreateFromD3D11Buffer>((cl_platform_id)platform, fname);
        }
    }

    BufferDX(
        const Context& context,
        cl_mem_flags flags,
        void* resource,
        cl_int * err = NULL) {
        cl_int error;
        ID3D11Buffer* buffer = static_cast<ID3D11Buffer*>(resource);
        object_ = pfn_clCreateFromD3D11Buffer(
            context(),
            flags,
            buffer,
            &error);

        detail::errHandler(error);
        if (err != NULL) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to NULL.
    BufferDX() : Buffer() { }

    /*! \brief Constructor from cl_mem - takes ownership.
    *
    * \param retainObject will cause the constructor to retain its cl object.
    *                     Defaults to false to maintain compatibility with
    *                     earlier versions.
    *  See Memory for further details.
    */
    explicit BufferDX(const cl_mem& buf, bool retainObject = false) :
        Buffer(buf, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
    *
    *  See Memory for further details.
    */
    BufferDX& operator = (const cl_mem& rhs) {
        Buffer::operator=(rhs);
        return *this;
    }

    /*! \brief Copy constructor to forward copy to the superclass correctly.
    * Required for MSVC.
    */
    BufferDX(const BufferDX& buf) :
        Buffer(buf) {}

    /*! \brief Copy assignment to forward copy to the superclass correctly.
    * Required for MSVC.
    */
    BufferDX& operator = (const BufferDX &buf) {
        Buffer::operator=(buf);
        return *this;
    }

    /*! \brief Move constructor to forward move to the superclass correctly.
    * Required for MSVC.
    */
    BufferDX(BufferDX&& buf) noexcept : Buffer(std::move(buf)) {}

    /*! \brief Move assignment to forward move to the superclass correctly.
    * Required for MSVC.
    */
    BufferDX& operator = (BufferDX &&buf) {
        Buffer::operator=(std::move(buf));
        return *this;
    }
};
#endif

class PlatformVA : public Platform {
public:
    //! \brief Default constructor - initializes to NULL.
    PlatformVA() : Platform() { }

    explicit PlatformVA(const cl_platform_id &platform, bool retainObject = false) :
        Platform(platform, retainObject) { }

    cl_int getDevices(
        cl_device_source_intel media_adapter_type,
        void *                 media_adapter,
        cl_device_set_intel    media_adapter_set,
        vector<Device>* devices) const {
        typedef CL_API_ENTRY cl_int(CL_API_CALL * PFN_clGetDeviceIDsFromMediaAdapterINTEL)(
            cl_platform_id /* platform */,
            cl_device_source_intel /* media_adapter_type */,
            void * /* media_adapter */,
            cl_device_set_intel /* media_adapter_set */,
            cl_uint /* num_entries */,
            cl_device_id * /* devices */,
            cl_uint * /* num_devices */);
#ifdef WIN32
        const char* fname = "clGetDeviceIDsFromD3D11KHR";
#else
        const char* fname = "clGetDeviceIDsFromVA_APIMediaAdapterINTEL";
#endif
        if (devices == NULL) {
            return detail::errHandler(CL_INVALID_ARG_VALUE, fname);
        }

        PFN_clGetDeviceIDsFromMediaAdapterINTEL pfn_clGetDeviceIDsFromMediaAdapterINTEL =
            try_load_entrypoint<PFN_clGetDeviceIDsFromMediaAdapterINTEL>(object_, fname);

        if (NULL == pfn_clGetDeviceIDsFromMediaAdapterINTEL) {
            return CL_INVALID_PLATFORM;
        }

        cl_uint n = 0;
        cl_int err = pfn_clGetDeviceIDsFromMediaAdapterINTEL(
            object_,
            media_adapter_type,
            media_adapter,
            media_adapter_set,
            0,
            NULL,
            &n);
        if (err != CL_SUCCESS && err != CL_DEVICE_NOT_FOUND) {
            return detail::errHandler(err, fname);
        }

        if (err != CL_DEVICE_NOT_FOUND) {
            vector<cl_device_id> ids(n);
            err = pfn_clGetDeviceIDsFromMediaAdapterINTEL(
                object_,
                media_adapter_type,
                media_adapter,
                media_adapter_set,
                n,
                ids.data(),
                NULL);
            if (err != CL_SUCCESS) {
                return detail::errHandler(err, fname);
            }

            // Cannot trivially assign because we need to capture intermediates
            // with safe construction
            // We must retain things we obtain from the API to avoid releasing
            // API-owned objects.
            if (devices) {
                devices->resize(ids.size());

                // Assign to param, constructing with retain behaviour
                // to correctly capture each underlying CL object
                for (size_type i = 0; i < ids.size(); i++) {
                    (*devices)[i] = Device(ids[i], true);
                }
            }

            // set up acquire/release extensions
            SharedSurfLock::Init(object_);
            ImageVA::Init(object_);
#ifdef WIN32
            BufferDX::Init(object_);
#endif
        }
        return CL_SUCCESS;
    }
};

class UsmHelper {
public:
    explicit UsmHelper(const cl::Context& ctx, const cl::Device device, bool use_usm) : _ctx(ctx), _device(device) {
        if (use_usm) {
            _host_mem_alloc_fn             = try_load_entrypoint<clHostMemAllocINTEL_fn>(_ctx.get(), "clHostMemAllocINTEL");
            _shared_mem_alloc_fn           = try_load_entrypoint<clSharedMemAllocINTEL_fn>(_ctx.get(), "clSharedMemAllocINTEL");
            _device_mem_alloc_fn           = try_load_entrypoint<clDeviceMemAllocINTEL_fn>(_ctx.get(), "clDeviceMemAllocINTEL");
            _mem_free_fn                   = try_load_entrypoint<clMemFreeINTEL_fn>(_ctx.get(), "clMemFreeINTEL");
            _set_kernel_arg_mem_pointer_fn = try_load_entrypoint<clSetKernelArgMemPointerINTEL_fn>(_ctx.get(), "clSetKernelArgMemPointerINTEL");
            _enqueue_memcpy_fn             = try_load_entrypoint<clEnqueueMemcpyINTEL_fn>(_ctx.get(), "clEnqueueMemcpyINTEL");
            _enqueue_mem_fill_fn           = try_load_entrypoint<clEnqueueMemFillINTEL_fn>(_ctx.get(), "clEnqueueMemFillINTEL");
            _enqueue_memset_fn             = try_load_entrypoint<clEnqueueMemsetINTEL_fn>(_ctx.get(), "clEnqueueMemsetINTEL");
            _get_mem_alloc_info_fn         = try_load_entrypoint<clGetMemAllocInfoINTEL_fn>(_ctx.get(), "clGetMemAllocInfoINTEL");
        }
    }

    void* allocate_host(const cl_mem_properties_intel *properties, size_t size, cl_uint alignment, cl_int* err_code_ret) const {\
        if (!_host_mem_alloc_fn)
            throw std::runtime_error("[CLDNN] clHostMemAllocINTEL is nullptr");
        return _host_mem_alloc_fn(_ctx.get(), properties, size, alignment, err_code_ret);
    }

    void* allocate_shared(const cl_mem_properties_intel *properties, size_t size, cl_uint alignment, cl_int* err_code_ret) const {
        if (!_shared_mem_alloc_fn)
            throw std::runtime_error("[CLDNN] clSharedMemAllocINTEL is nullptr");
        return _shared_mem_alloc_fn(_ctx.get(), _device.get(), properties, size, alignment, err_code_ret);
    }

    void* allocate_device(const cl_mem_properties_intel *properties, size_t size, cl_uint alignment, cl_int* err_code_ret) const {
        if (!_device_mem_alloc_fn)
            throw std::runtime_error("[CLDNN] clDeviceMemAllocINTEL is nullptr");
        return _device_mem_alloc_fn(_ctx.get(), _device.get(), properties, size, alignment, err_code_ret);
    }

    void free_mem(void* ptr) const {
        if (!_mem_free_fn)
            throw std::runtime_error("[CLDNN] clMemFreeINTEL is nullptr");
        _mem_free_fn(_ctx.get(), ptr);
    }

    cl_int set_kernel_arg_mem_pointer(const cl::Kernel& kernel, uint32_t index, const void* ptr) const {
        if (!_set_kernel_arg_mem_pointer_fn)
            throw std::runtime_error("[CLDNN] clSetKernelArgMemPointerINTEL is nullptr");
        return _set_kernel_arg_mem_pointer_fn(kernel.get(), index, ptr);
    }

    cl_int enqueue_memcpy(const cl::CommandQueue& cpp_queue, void *dst_ptr, const void *src_ptr,
                          size_t bytes_count, bool blocking = true, const std::vector<cl::Event>* wait_list = nullptr, cl::Event* ret_event = nullptr) const {
        if (!_enqueue_memcpy_fn)
            throw std::runtime_error("[CLDNN] clEnqueueMemcpyINTEL is nullptr");
        cl_event tmp;
        cl_int err = _enqueue_memcpy_fn(
            cpp_queue.get(),
            static_cast<cl_bool>(blocking),
            dst_ptr,
            src_ptr,
            bytes_count,
            wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
            wait_list == nullptr ? nullptr : reinterpret_cast<const cl_event*>(&wait_list->front()),
            ret_event == nullptr ? nullptr : &tmp);

        if (ret_event != nullptr && err == CL_SUCCESS)
            *ret_event = tmp;

        return err;
    }

    cl_int enqueue_fill_mem(const cl::CommandQueue& cpp_queue, void *dst_ptr, const void* pattern,
                            size_t pattern_size, size_t bytes_count, const std::vector<cl::Event>* wait_list = nullptr,
                            cl::Event* ret_event = nullptr) const {
        if (!_enqueue_mem_fill_fn)
            throw std::runtime_error("[CLDNN] clEnqueueMemFillINTEL is nullptr");
        cl_event tmp;
        cl_int err = _enqueue_mem_fill_fn(
            cpp_queue.get(),
            dst_ptr,
            pattern,
            pattern_size,
            bytes_count,
            wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
            wait_list == nullptr ? nullptr :  reinterpret_cast<const cl_event*>(&wait_list->front()),
            ret_event == nullptr ? nullptr : &tmp);

        if (ret_event != nullptr && err == CL_SUCCESS)
            *ret_event = tmp;

        return err;
    }

    cl_int enqueue_set_mem(const cl::CommandQueue& cpp_queue, void* dst_ptr, cl_int value,
                           size_t bytes_count, const std::vector<cl::Event>* wait_list = nullptr,
                           cl::Event* ret_event = nullptr) const {
        if (!_enqueue_memset_fn)
            throw std::runtime_error("[CLDNN] clEnqueueMemsetINTEL is nullptr");
        cl_event tmp;
        cl_int err = _enqueue_memset_fn(
            cpp_queue.get(),
            dst_ptr,
            value,
            bytes_count,
            wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
            wait_list == nullptr ? nullptr :  reinterpret_cast<const cl_event*>(&wait_list->front()),
            ret_event == nullptr ? nullptr : &tmp);

        if (ret_event != nullptr && err == CL_SUCCESS)
            *ret_event = tmp;

        return err;
    }

    cl_unified_shared_memory_type_intel get_usm_allocation_type(const void* usm_ptr) const {
        if (!_get_mem_alloc_info_fn) {
            throw std::runtime_error("[GPU] clGetMemAllocInfoINTEL is nullptr");
        }

        cl_unified_shared_memory_type_intel ret_val;
        size_t ret_val_size;
        _get_mem_alloc_info_fn(_ctx.get(), usm_ptr, CL_MEM_ALLOC_TYPE_INTEL, sizeof(cl_unified_shared_memory_type_intel), &ret_val, &ret_val_size);
        return ret_val;
    }

    size_t get_usm_allocation_size(const void* usm_ptr) const {
        if (!_get_mem_alloc_info_fn) {
            throw std::runtime_error("[GPU] clGetMemAllocInfoINTEL is nullptr");
        }

        size_t ret_val;
        size_t ret_val_size;
        _get_mem_alloc_info_fn(_ctx.get(), usm_ptr, CL_MEM_ALLOC_SIZE_INTEL, sizeof(size_t), &ret_val, &ret_val_size);
        return ret_val;
    }

private:
    cl::Context _ctx;
    cl::Device _device;
    clHostMemAllocINTEL_fn _host_mem_alloc_fn = nullptr;
    clMemFreeINTEL_fn _mem_free_fn = nullptr;
    clSharedMemAllocINTEL_fn _shared_mem_alloc_fn = nullptr;
    clDeviceMemAllocINTEL_fn _device_mem_alloc_fn = nullptr;
    clSetKernelArgMemPointerINTEL_fn _set_kernel_arg_mem_pointer_fn = nullptr;
    clEnqueueMemcpyINTEL_fn _enqueue_memcpy_fn = nullptr;
    clEnqueueMemFillINTEL_fn _enqueue_mem_fill_fn = nullptr;
    clEnqueueMemsetINTEL_fn _enqueue_memset_fn = nullptr;
    clGetMemAllocInfoINTEL_fn _get_mem_alloc_info_fn = nullptr;
};

/*
    UsmPointer requires associated context to free it.
    Simple wrapper class for usm allocated pointer.
*/
class UsmHolder {
public:
    UsmHolder(const cl::UsmHelper& usmHelper, void* ptr, bool shared_memory = false)
    : _usmHelper(usmHelper)
    , _ptr(ptr)
    , _shared_memory(shared_memory) { }

    void* ptr() { return _ptr; }
    void memFree() {
        try {
            if (!_shared_memory)
                _usmHelper.free_mem(_ptr);
        } catch (...) {
            // Exception may happen only when clMemFreeINTEL function is unavailable, thus can't free memory properly
        }
        _ptr = nullptr;
    }
    ~UsmHolder() {
        memFree();
    }
private:
    const cl::UsmHelper& _usmHelper;
    void* _ptr;
    bool _shared_memory = false;
};
/*
    USM base class. Different usm types should derive from this class.
*/
class UsmMemory {
public:
    explicit UsmMemory(const cl::UsmHelper& usmHelper) : _usmHelper(usmHelper) { }
    UsmMemory(const cl::UsmHelper& usmHelper, void* usm_ptr, size_t offset = 0)
    : _usmHelper(usmHelper)
    , _usm_pointer(std::make_shared<UsmHolder>(_usmHelper, reinterpret_cast<uint8_t*>(usm_ptr) + offset, true)) {
        if (!usm_ptr) {
            throw std::runtime_error("[GPU] Can't share null usm pointer");
        }
    }

    // Get methods returns original pointer allocated by openCL.
    void* get() const { return _usm_pointer->ptr(); }

    void allocateHost(size_t size) {
        cl_int error = CL_SUCCESS;
        auto ptr = _usmHelper.allocate_host(nullptr, size, 0, &error);
        _check_error(size, ptr, error, "Host");
        _allocate(ptr);
    }

    void allocateShared(size_t size) {
        cl_int error = CL_SUCCESS;
        auto ptr = _usmHelper.allocate_shared(nullptr, size, 0, &error);
        _check_error(size, ptr, error, "Shared");
        _allocate(ptr);
    }

    void allocateDevice(size_t size) {
        cl_int error = CL_SUCCESS;
        auto ptr = _usmHelper.allocate_device(nullptr, size, 0, &error);
        _check_error(size, ptr, error, "Device");
        _allocate(ptr);
    }

    void freeMem() {
        if (!_usm_pointer)
            throw std::runtime_error("[CL ext] Can not free memory of empty UsmHolder");
        _usm_pointer->memFree();
    }

    virtual ~UsmMemory() = default;

protected:
    const UsmHelper& _usmHelper;
    std::shared_ptr<UsmHolder> _usm_pointer = nullptr;

private:
    void _allocate(void* ptr) {
        _usm_pointer = std::make_shared<UsmHolder>(_usmHelper, ptr);
    }

    void _check_error(size_t size, void* ptr, cl_int error, const char* usm_type) {
        if (ptr == nullptr || error != CL_SUCCESS) {
            std::stringstream sout;
            sout << "[CL ext] Can not allocate " << size << " bytes for USM " << usm_type << ". ptr: " << ptr << ", error: " << error << std::endl;
            if (ptr == nullptr)
                throw std::runtime_error(sout.str());
            else
                detail::errHandler(error, sout.str().c_str());
        }
    }
};

/*
    Wrapper for standard cl::Kernel object.
    Extend cl::Kernel functionality.
*/
class KernelIntel : public Kernel {
    using Kernel::Kernel;
public:
    explicit KernelIntel(const UsmHelper& usmHelper) : Kernel(), _usmHelper(usmHelper) {}
    KernelIntel(const Kernel &other, const UsmHelper& usmHelper) : Kernel(other), _usmHelper(usmHelper) { }

    KernelIntel clone() const {
        Kernel cloned_kernel(this->getInfo<CL_KERNEL_PROGRAM>(), this->getInfo<CL_KERNEL_FUNCTION_NAME>().c_str());
        return KernelIntel(cloned_kernel, _usmHelper);
    }

    cl_int setArgUsm(cl_uint index, const UsmMemory& mem) {
        return detail::errHandler(_usmHelper.set_kernel_arg_mem_pointer(*this, index, mem.get()), "[CL_EXT] setArgUsm in KernelIntel failed");
    }
private:
    const UsmHelper& _usmHelper;
};

inline bool operator==(const UsmMemory &lhs, const UsmMemory &rhs) {
    return lhs.get() == rhs.get();
}

inline bool operator!=(const UsmMemory &lhs, const UsmMemory &rhs) {
    return !operator==(lhs, rhs);
}

}  //namespace cl
