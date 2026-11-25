#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cstddef>

#include "exportApi.h"
#include "IMegakernelRuntime.h"

namespace mk {
struct ConstantParamsImpl : IConstantParams {};
struct RuntimeParamsImpl : IRuntimeParams {
    void* left = nullptr;
    void* right = nullptr;
    void* output = nullptr;
    size_t count = 0;
};
struct PlatformParamsImpl : IPlatformParams {
    cl_device_id deviceId = nullptr;
    cl_context context = nullptr;
    cl_command_queue stream = nullptr;
};
}

extern "C" TEST_MEGAKERNEL_EXPORT mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();
extern "C" TEST_MEGAKERNEL_EXPORT void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime);
extern "C" TEST_MEGAKERNEL_EXPORT unsigned int TestMegaKernelExecutionCount();