// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


///
/// \file This file wraps cl2.hpp and introduces wrapper classes for Intel sharing extensions.
///

#pragma once
#include "cl2.hpp"

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
#include <CL/cl_intel_planar_yuv.h>

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

            static PFN_clGetDeviceIDsFromMediaAdapterINTEL pfn_clGetDeviceIDsFromMediaAdapterINTEL = NULL;
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
            if (err != CL_SUCCESS) {
                return detail::errHandler(err, fname);
            }

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

            return CL_SUCCESS;
        }
    };
}
