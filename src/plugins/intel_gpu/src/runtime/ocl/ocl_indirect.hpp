#pragma once

#ifdef _WIN32
# ifndef NOMINMAX
#  define NOMINMAX
# endif
#endif

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#endif

#include <string>
#include <stdexcept>

class opencl_error : public std::runtime_error {
public:
    opencl_error(cl_int status_ = 0) : std::runtime_error("An OpenCL error occurred: " + std::to_string(status_)), status(status_) {}
protected:
    cl_int status;
};

// Dynamically loaded level_zero functions
namespace {

void *find_cl_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libOpencl.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    // Use LOAD_LIBRARY_SEARCH_SYSTEM32 flag to avoid DLL hijacking issue.
    HMODULE handle = LoadLibraryExA(
            "OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#endif
    if (!handle) throw opencl_error();

#if defined(__linux__)
    void *f = reinterpret_cast<void *>(dlsym(handle, symbol));
#elif defined(_WIN32)
    void *f = reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#endif

    if (!f) throw opencl_error();
    return f;
}

template <typename F>
F find_cl_symbol(const char *symbol) {
    return (F)find_cl_symbol(symbol);
}

#define CL_INDIRECT_API(f) \
    template <typename... Args> auto call_##f(Args&&... args) { \
        static auto f_ = find_cl_symbol<decltype(&f)>(#f);              \
        return f_(std::forward<Args>(args)...);                         \
    }

CL_INDIRECT_API(clBuildProgram)
CL_INDIRECT_API(clCreateBuffer)
CL_INDIRECT_API(clCreateContext)
CL_INDIRECT_API(clCreateKernel)
CL_INDIRECT_API(clCreateProgramWithBinary)
CL_INDIRECT_API(clCreateProgramWithSource)
CL_INDIRECT_API(clCreateSubBuffer)
CL_INDIRECT_API(clCreateSubDevices)
CL_INDIRECT_API(clEnqueueMapBuffer)
CL_INDIRECT_API(clEnqueueUnmapMemObject)
CL_INDIRECT_API(clFinish)
CL_INDIRECT_API(clGetDeviceIDs)
CL_INDIRECT_API(clGetExtensionFunctionAddressForPlatform)
CL_INDIRECT_API(clGetKernelArgInfo)
CL_INDIRECT_API(clGetMemObjectInfo)
CL_INDIRECT_API(clGetPlatformIDs)
CL_INDIRECT_API(clGetPlatformInfo)
CL_INDIRECT_API(clGetProgramInfo)
CL_INDIRECT_API(clReleaseCommandQueue)
CL_INDIRECT_API(clReleaseContext)
CL_INDIRECT_API(clReleaseDevice)
CL_INDIRECT_API(clReleaseEvent)
CL_INDIRECT_API(clReleaseKernel)
CL_INDIRECT_API(clReleaseMemObject)
CL_INDIRECT_API(clReleaseProgram)
CL_INDIRECT_API(clReleaseSampler)
CL_INDIRECT_API(clRetainCommandQueue)
CL_INDIRECT_API(clRetainContext)
CL_INDIRECT_API(clRetainDevice)
CL_INDIRECT_API(clRetainEvent)
CL_INDIRECT_API(clRetainKernel)
CL_INDIRECT_API(clRetainMemObject)
CL_INDIRECT_API(clRetainProgram)
CL_INDIRECT_API(clRetainSampler)
CL_INDIRECT_API(clCreateCommandQueueWithProperties)
CL_INDIRECT_API(clCreateCommandQueue)
CL_INDIRECT_API(clWaitForEvents)
CL_INDIRECT_API(clEnqueueBarrierWithWaitList)
CL_INDIRECT_API(clEnqueueMarkerWithWaitList)
CL_INDIRECT_API(clEnqueueNDRangeKernel)
CL_INDIRECT_API(clEnqueueReadBuffer)
CL_INDIRECT_API(clEnqueueWriteBuffer)
CL_INDIRECT_API(clEnqueueFillBuffer)
CL_INDIRECT_API(clEnqueueCopyBuffer)
CL_INDIRECT_API(clEnqueueFillImage)
CL_INDIRECT_API(clEnqueueMapImage)
CL_INDIRECT_API(clEnqueueWriteImage)
CL_INDIRECT_API(clEnqueueCopyImage)
CL_INDIRECT_API(clFlush)
CL_INDIRECT_API(clSetUserEventStatus)
CL_INDIRECT_API(clSetEventCallback)
CL_INDIRECT_API(clUnloadPlatformCompiler);
CL_INDIRECT_API(clCreateUserEvent);
CL_INDIRECT_API(clCreateBufferWithProperties);
CL_INDIRECT_API(clCreateContextFromType);
CL_INDIRECT_API(clSetKernelArg);

cl_int call_clGetEventProfilingInfo(cl_event a1, cl_profiling_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetEventProfilingInfo)>("clGetEventProfilingInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }
cl_int call_clGetImageInfo(cl_mem a1, cl_image_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetImageInfo)>("clGetImageInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }
cl_int call_clGetMemObjectInfo(cl_mem a1, cl_mem_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetMemObjectInfo)>("clGetMemObjectInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }
cl_int call_clGetEventInfo(cl_event a1, cl_event_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetEventInfo)>("clGetEventInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }

cl_int call_clGetDeviceInfo(cl_device_id a1, cl_device_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetDeviceInfo)>("clGetDeviceInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }
    
cl_int call_clCreateKernelsInProgram(cl_program a1, cl_uint a2, cl_kernel* a3, cl_uint* a4) { \
        static auto f_ = find_cl_symbol<decltype(&clCreateKernelsInProgram)>("clCreateKernelsInProgram");              \
        return f_(a1, a2, a3, a4);                         \
    }

cl_int call_clGetPlatformInfo(cl_platform_id a1, cl_platform_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetPlatformInfo)>("clGetPlatformInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }
cl_int call_clGetKernelInfo(cl_kernel a1, cl_kernel_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetKernelInfo)>("clGetKernelInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }
cl_int call_clGetProgramBuildInfo(cl_program a1, cl_device_id a2, cl_program_build_info a3, size_t a4, void* a5, size_t* a6) { \
        static auto f_ = find_cl_symbol<decltype(&clGetProgramBuildInfo)>("clGetProgramBuildInfo");              \
        return f_(a1, a2, a3, a4, a5, a6);                         \
    }
    
cl_int call_clGetContextInfo(cl_context a1, cl_context_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetContextInfo)>("clGetContextInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }

cl_int call_clGetCommandQueueInfo(cl_command_queue a1, cl_command_queue_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetCommandQueueInfo)>("clGetCommandQueueInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }
#undef CL_INDIRECT_API
} // namespace
