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
CL_INDIRECT_API(clGetContextInfo)
CL_INDIRECT_API(clGetDeviceIDs)
CL_INDIRECT_API(clGetDeviceInfo)
CL_INDIRECT_API(clGetExtensionFunctionAddressForPlatform)
CL_INDIRECT_API(clGetKernelArgInfo)
CL_INDIRECT_API(clGetKernelInfo)
CL_INDIRECT_API(clGetMemObjectInfo)
CL_INDIRECT_API(clGetPlatformIDs)
CL_INDIRECT_API(clGetPlatformInfo)
CL_INDIRECT_API(clGetProgramBuildInfo)
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
CL_INDIRECT_API(clGetCommandQueueInfo)
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

cl_int call_clGetEventProfilingInfo(cl_event a1, cl_profiling_info a2, size_t a3, void* a4, size_t* a5) { \
        static auto f_ = find_cl_symbol<decltype(&clGetEventProfilingInfo)>("clGetEventProfilingInfo");              \
        return f_(a1, a2, a3, a4, a5);                         \
    }


#undef CL_INDIRECT_API
} // namespace