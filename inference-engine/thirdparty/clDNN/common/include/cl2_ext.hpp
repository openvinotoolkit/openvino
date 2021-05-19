// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// \file This file wraps cl2.hpp and introduces wrapper classes for Intel sharing extensions.
///

#pragma once
#include <CL/opencl.hpp>
#define NOMINMAX
#ifdef _WIN32
#include <CL/cl_d3d11.h>
typedef cl_d3d11_device_source_khr cl_device_source_intel;
typedef cl_d3d11_device_set_khr    cl_device_set_intel;
#else
#include <CL/cl_va_api_media_sharing_intel.h>
typedef cl_va_api_device_source_intel cl_device_source_intel;
typedef cl_va_api_device_set_intel    cl_device_set_intel;
#endif

// cl_intel_device_attribute_query
#define CL_DEVICE_IP_VERSION_INTEL                0x4250
#define CL_DEVICE_ID_INTEL                        0x4251
#define CL_DEVICE_NUM_SLICES_INTEL                0x4252
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL  0x4253
#define CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL     0x4254
#define CL_DEVICE_NUM_THREADS_PER_EU_INTEL        0x4255
#define CL_DEVICE_FEATURE_CAPABILITIES_INTEL      0x4256

typedef cl_bitfield         cl_device_feature_capabilities_intel;

/* For GPU devices, version 1.0.0: */

#define CL_DEVICE_FEATURE_FLAG_DP4A_INTEL         (1 << 0)


namespace cl {
namespace detail {
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_IP_VERSION_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_ID_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_SLICES_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_THREADS_PER_EU_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_FEATURE_CAPABILITIES_INTEL, cl_device_feature_capabilities_intel)
}
}

#include <memory>

namespace {
template <typename T>
T load_entrypoint(const cl_platform_id platform, const std::string name) {
    T p = reinterpret_cast<T>(
        clGetExtensionFunctionAddressForPlatform(platform, name.c_str()));
    if (!p) {
        throw std::runtime_error("clGetExtensionFunctionAddressForPlatform(" +
            name + ") returned NULL.");
    }
    return p;
}

template <typename T>
T load_entrypoint(const cl_device_id device, const std::string name) {
    cl_platform_id platform;
    cl_int error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
        &platform, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_DEVICE_PLATFORM: " +
            std::to_string(error));
    }
    return load_entrypoint<T>(platform, name);
}

// loader functions created for single device contexts
// ToDo Extend it for multi device case.
template <typename T>
T load_entrypoint(const cl_context context, const std::string name) {
    size_t size = 0;
    cl_int error =
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &size);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_CONTEXT_DEVICES size: " +
            std::to_string(error));
    }

    std::vector<cl_device_id> devices(size / sizeof(cl_device_id));

    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices.data(),
        nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_CONTEXT_DEVICES: " +
            std::to_string(error));
    }

    return load_entrypoint<T>(devices.front(), name);
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
} // namespace


namespace cl {
namespace usm {
/*
A static local variable in an extern inline function always refers to the same object.
7.1.2/4 - C++98/C++14 (n3797)
+ Functions are by default extern, unless specified as static.
*/
inline void* hostMemAlloc(const cl::Context& cpp_context, const cl_mem_properties_intel *properties, size_t size,
    cl_uint alignment, cl_int* err_code_ret) {
    clHostMemAllocINTEL_fn fn = load_entrypoint<clHostMemAllocINTEL_fn>(cpp_context.get(), "clHostMemAllocINTEL");
    return fn(cpp_context.get(), properties, size, alignment, err_code_ret);
}

inline void* sharedMemAlloc(const cl::Device& cpp_device, const cl::Context& cpp_context, const cl_mem_properties_intel *properties, size_t size,
    cl_uint alignment, cl_int* err_code_ret) {
    clSharedMemAllocINTEL_fn fn = load_entrypoint<clSharedMemAllocINTEL_fn>(cpp_context.get(), "clSharedMemAllocINTEL");
    return fn(cpp_context.get(), cpp_device.get(), properties, size, alignment, err_code_ret);
}

inline void* deviceMemAlloc(const cl::Device& cpp_device, const cl::Context& cpp_context, const cl_mem_properties_intel *properties, size_t size,
    cl_uint alignment, cl_int* err_code_ret) {
    clDeviceMemAllocINTEL_fn fn = load_entrypoint<clDeviceMemAllocINTEL_fn>(cpp_context.get(), "clDeviceMemAllocINTEL");
    return fn(cpp_context.get(), cpp_device.get(), properties, size, alignment, err_code_ret);
}

inline cl_int memFree(const cl::Context& cpp_context, void* ptr) {
    clMemFreeINTEL_fn fn = load_entrypoint<clMemFreeINTEL_fn>(cpp_context.get(), "clMemFreeINTEL");
    return fn(cpp_context.get(), ptr);
}

inline cl_int set_kernel_arg_mem_pointer(const cl::Kernel& kernel, uint32_t index, const void* ptr, clSetKernelArgMemPointerINTEL_fn fn) {
    return fn(kernel.get(), index, ptr);
}

inline cl_int enqueue_memcpy(const cl::CommandQueue& cpp_queue, void *dst_ptr, const void *src_ptr,
    size_t bytes_count, bool blocking = true, const std::vector<cl::Event>* wait_list = nullptr, cl::Event* ret_event = nullptr) {
    clEnqueueMemcpyINTEL_fn fn = load_entrypoint<clEnqueueMemcpyINTEL_fn>(cpp_queue.get(), "clEnqueueMemcpyINTEL");

    cl_event tmp;
    cl_int err = fn(
        cpp_queue.get(),
        static_cast<cl_bool>(blocking),
        dst_ptr,
        src_ptr,
        bytes_count,
        wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
        wait_list == nullptr ? nullptr : (cl_event*)&wait_list->front(),
        ret_event == nullptr ? nullptr : &tmp);

    if (ret_event != nullptr && err == CL_SUCCESS)
        *ret_event = tmp;

    return err;
}

inline cl_int enqueue_fill_mem(const cl::CommandQueue& cpp_queue, void *dst_ptr, const void* pattern,
    size_t pattern_size, size_t bytes_count, const std::vector<cl::Event>* wait_list = nullptr,
    cl::Event* ret_event = nullptr) {
    clEnqueueMemFillINTEL_fn fn = load_entrypoint<clEnqueueMemFillINTEL_fn>(cpp_queue.get(), "clEnqueueMemFillINTEL");

    cl_event tmp;
    cl_int err = fn(
        cpp_queue.get(),
        dst_ptr,
        pattern,
        pattern_size,
        bytes_count,
        wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
        wait_list == nullptr ? nullptr : (cl_event*)&wait_list->front(),
        ret_event == nullptr ? nullptr : &tmp);

    if (ret_event != nullptr && err == CL_SUCCESS)
        *ret_event = tmp;

    return err;
}

inline cl_int enqueue_set_mem(const cl::CommandQueue& cpp_queue, void* dst_ptr, cl_int value,
    size_t bytes_count, const std::vector<cl::Event>* wait_list = nullptr,
    cl::Event* ret_event = nullptr) {
    clEnqueueMemsetINTEL_fn fn = load_entrypoint<clEnqueueMemsetINTEL_fn>(cpp_queue.get(), "clEnqueueMemsetINTEL");

    cl_event tmp;
    cl_int err = fn(
        cpp_queue.get(),
        dst_ptr,
        value,
        bytes_count,
        wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
        wait_list == nullptr ? nullptr : (cl_event*)&wait_list->front(),
        ret_event == nullptr ? nullptr : &tmp);

    if (ret_event != nullptr && err == CL_SUCCESS)
        *ret_event = tmp;

    return err;
}
} //namespace usm

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
                pfn_acquire =
                    reinterpret_cast<PFN_clEnqueueAcquireMediaSurfacesINTEL>
                    (clGetExtensionFunctionAddressForPlatform(platform, fnameAcq));
            }
            if (!pfn_release) {
                pfn_release = reinterpret_cast<PFN_clEnqueueReleaseMediaSurfacesINTEL>
                (clGetExtensionFunctionAddressForPlatform(platform, fnameRel));
            }
        }

        SharedSurfLock(cl_command_queue queue,
            std::vector<cl_mem>& surfaces,
            cl_int * err = NULL)
            : m_queue(queue), m_surfaces(surfaces), m_errPtr(err) {

            if (pfn_acquire!=NULL && m_surfaces.size()) {
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

    class ImageVA : public Image2D
    {
    public:
        static PFN_clCreateFromMediaSurfaceINTEL pfn_clCreateFromMediaSurfaceINTEL;

        static void Init(cl_platform_id platform) {
#ifdef WIN32
            const char* fname = "clCreateFromD3D11Texture2DKHR";
#else
            const char* fname = "clCreateFromVA_APIMediaSurfaceINTEL";
#endif
            if (!pfn_clCreateFromMediaSurfaceINTEL) {
                pfn_clCreateFromMediaSurfaceINTEL =
                    reinterpret_cast<PFN_clCreateFromMediaSurfaceINTEL>
                    (clGetExtensionFunctionAddressForPlatform((cl_platform_id)platform, fname));
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
            cl_int * err = NULL
        )
        {
            cl_int error;
            object_ = pfn_clCreateFromMediaSurfaceINTEL(
                context(),
                flags,
#ifdef WIN32
                surface,
#else
                (void*)& surface,
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
        ImageVA& operator = (const cl_mem& rhs)
        {
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
        ImageVA& operator = (const ImageVA &img)
        {
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
        ImageVA& operator = (ImageVA &&buf)
        {
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
                pfn_clCreateFromD3D11Buffer = reinterpret_cast<PFN_clCreateFromD3D11Buffer>
                    (clGetExtensionFunctionAddressForPlatform((cl_platform_id)platform, fname));
            }
        }

        BufferDX(
            const Context& context,
            cl_mem_flags flags,
            void* resource,
            cl_int * err = NULL
        )
        {
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
        BufferDX& operator = (const cl_mem& rhs)
        {
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
        BufferDX& operator = (const BufferDX &buf)
        {
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
        BufferDX& operator = (BufferDX &&buf)
        {
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
            vector<Device>* devices) const
        {
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

            PFN_clGetDeviceIDsFromMediaAdapterINTEL pfn_clGetDeviceIDsFromMediaAdapterINTEL = NULL;
            if (!pfn_clGetDeviceIDsFromMediaAdapterINTEL) {
                pfn_clGetDeviceIDsFromMediaAdapterINTEL =
                    reinterpret_cast<PFN_clGetDeviceIDsFromMediaAdapterINTEL>
                    (clGetExtensionFunctionAddressForPlatform(object_, fname));

                if (NULL == pfn_clGetDeviceIDsFromMediaAdapterINTEL) {
                    return CL_INVALID_PLATFORM;
                }
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

            if (err != CL_DEVICE_NOT_FOUND)
            {
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

    /*
        UsmPointer requires associated context to free it.
        Simple wrapper class for usm allocated pointer.
    */
    class UsmHolder {
    public:
        explicit UsmHolder(Context& ctx, void* ptr) : _ctx(ctx), _ptr(ptr) {}
        void* ptr() { return _ptr; }
        ~UsmHolder() { usm::memFree(_ctx, _ptr); }
    private:
        Context _ctx;
        void* _ptr;
    };

    /*
        USM base class. Different usm types should derive from this class.
    */
    class UsmMemory {
    public:
        UsmMemory() = default;

        explicit UsmMemory(const Context& ctx)
            : _ctx(ctx) {
        }

        // Get methods returns original pointer allocated by openCL.
        void* get() const { return _usm_pointer->ptr(); }
        size_t use_count() const { return _usm_pointer.use_count(); }

        void allocateHost(size_t size) {
            cl_int error = CL_SUCCESS;
            _allocate(usm::hostMemAlloc(_ctx, nullptr, size, 0, &error));
            if (error != CL_SUCCESS)
                detail::errHandler(error, "[CL_EXT] UsmHost in cl extensions constructor failed");
        }

        void allocateShared(const cl::Device& device, size_t size) {
            cl_int error = CL_SUCCESS;
            _allocate(usm::sharedMemAlloc(device, _ctx, nullptr, size, 0, &error));
            if (error != CL_SUCCESS)
                detail::errHandler(error, "[CL_EXT] UsmShared in cl extensions constructor failed");
        }

        void allocateDevice(const cl::Device& device, size_t size) {
            cl_int error = CL_SUCCESS;
            _allocate(usm::deviceMemAlloc(device, _ctx, nullptr, size, 0, &error));
            if (error != CL_SUCCESS)
                detail::errHandler(error, "[CL_EXT] UsmDevice in cl extensions constructor failed");
        }

        virtual ~UsmMemory() = default;

    protected:
        std::shared_ptr<UsmHolder> _usm_pointer = nullptr;
        Context _ctx;

     private:
        void _allocate(void* ptr) {
            if (!ptr)
                throw std::runtime_error("[CL ext] Can not allocate nullptr for USM type.");
            _usm_pointer = std::make_shared<UsmHolder>(_ctx, ptr);
        }
    };

    /*
        Wrapper for standard cl::Kernel object.
        Extend cl::Kernel functionality.
    */
    class KernelIntel : public Kernel {
        using Kernel::Kernel;
        clSetKernelArgMemPointerINTEL_fn fn;

    public:
        KernelIntel() : Kernel(), fn(nullptr) {}
        KernelIntel(const Kernel &other, bool supports_usm) : Kernel(other) {
            fn = supports_usm ? load_entrypoint<clSetKernelArgMemPointerINTEL_fn>(get(), "clSetKernelArgMemPointerINTEL") : nullptr;
        }

        KernelIntel(const Kernel &other, clSetKernelArgMemPointerINTEL_fn fn) : Kernel(other), fn(fn) { }

        KernelIntel clone() const {
            Kernel cloned_kernel(this->getInfo<CL_KERNEL_PROGRAM>(), this->getInfo<CL_KERNEL_FUNCTION_NAME>().c_str());
            return KernelIntel(cloned_kernel, fn);
        }

        cl_int setArgUsm(cl_uint index, const UsmMemory& mem) {
            if (!fn)
                throw std::runtime_error("[CL ext] clSetKernelArgMemPointerINTEL function ptr is null. Can not set USM arg.");

            return detail::errHandler(
                usm::set_kernel_arg_mem_pointer(
                *this, index, mem.get(), fn),
                "[CL_EXT] setArgUsm in KernelIntel failed");
        }
    };

    /*
    Wrapper for standard cl::CommandQueue object.
    Extend cl::CommandQueue functionality.
    */
    class CommandQueueIntel : public CommandQueue {
        using CommandQueue::CommandQueue;

    public:
        CommandQueueIntel() : CommandQueue() {}
        explicit CommandQueueIntel(const CommandQueue &other) : CommandQueue(other) {}
        explicit CommandQueueIntel(const cl_command_queue &other) : CommandQueue(other) {}

        cl_int enqueueCopyUsm(
            const UsmMemory& src,
            UsmMemory& dst,
            size_type size,
            bool blocking = true,
            const vector<Event>* events = nullptr,
            Event* event = nullptr) const {
            cl_int err = detail::errHandler(usm::enqueue_memcpy(
                *this,
                dst.get(),
                src.get(),
                size,
                blocking,
                events,
                event),
                "[CL_EXT] USM enqueue copy failed.");
            return err;
        }

        template<typename PatternType>
        cl_int enqueueFillUsm(
            UsmMemory& dst,
            PatternType pattern,
            size_type size,
            const vector<Event>* events = nullptr,
            Event* event = nullptr) const {
            cl_int err = detail::errHandler(usm::enqueue_fill_mem(
                *this,
                dst.get(),
                static_cast<void*>(&pattern),
                sizeof(PatternType),
                size,
                events,
                event),
                "[CL_EXT] USM enqueue fill failed.");
            return err;
        }

        cl_int enqueueSetUsm(
            UsmMemory& dst,
            int32_t value,
            size_type size,
            const vector<Event>* events = nullptr,
            Event* event = nullptr) const {
            cl_int err = detail::errHandler(usm::enqueue_set_mem(
                *this,
                dst.get(),
                static_cast<cl_int>(value),
                size,
                events,
                event),
                "[CL_EXT] USM enqueue fill failed.");
            return err;
        }
    };

	inline bool operator==(const UsmMemory &lhs, const UsmMemory &rhs) {
		return lhs.get() == rhs.get();
	}

	inline bool operator!=(const UsmMemory &lhs, const UsmMemory &rhs) {
		return !operator==(lhs, rhs);
	}
}  //namespace cl
