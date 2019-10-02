/*****************************************************************************\

Copyright (c) 2013-2014 Intel Corporation All Rights Reserved.

THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

File Name: va_ext.h

Abstract:

Notes:

\*****************************************************************************/
#ifndef __OPENCL_VA_EXT_H
#define __OPENCL_VA_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <va/va.h>

//cl_va_api_device_source_intel
#define CL_VA_API_DISPLAY_INTEL                             0x4094

//cl_va_api_device_set_intel
#define CL_PREFERRED_DEVICES_FOR_VA_API_INTEL               0x4095
#define CL_ALL_DEVICES_FOR_VA_API_INTEL                     0x4096

// cl_context_info
#define CL_CONTEXT_VA_API_DISPLAY_INTEL                     0x4097

// cl_mem_info
#define CL_MEM_VA_API_MEDIA_SURFACE_INTEL                   0x4098
// cl_image_info
#define CL_IMAGE_VA_API_PLANE_INTEL                         0x4099

//error codes
#define CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL               -1098
#define CL_INVALID_VA_API_MEDIA_SURFACE_INTEL               -1099
#define CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL      -1100
#define CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL          -1101

// cl_command_type
#define CL_COMMAND_ACQUIRE_VA_API_MEDIA_SURFACES_INTEL       0x409A
#define CL_COMMAND_RELEASE_VA_API_MEDIA_SURFACES_INTEL       0x409B

typedef cl_uint cl_va_api_device_source_intel;
typedef cl_uint cl_va_api_device_set_intel;

typedef CL_API_ENTRY cl_int (CL_API_CALL * clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn)(
    cl_platform_id                platform,
    cl_va_api_device_source_intel media_adapter_type,
    void                          *media_adapter,
    cl_va_api_device_set_intel    media_adapter_set,
    cl_uint                       num_entries,
    cl_device_id                  *devices,
    cl_uint                       *num_devices ) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_mem (CL_API_CALL * clCreateFromVA_APIMediaSurfaceINTEL_fn)(
    cl_context context,
    cl_mem_flags flags,
    VASurfaceID *surface,
    cl_uint plane,
    cl_int *errcode_ret ) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem     *mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event   *event_wait_list,
    cl_event         *ocl_event ) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem     *mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event   *event_wait_list,
    cl_event         *ocl_event ) CL_EXT_SUFFIX__VERSION_1_2;

#ifdef __cplusplus
}
#endif

#endif
