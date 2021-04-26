// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_kernel.hpp"
#include "sycl_engine.hpp"
#include "kernels_factory.hpp"

#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

#include <memory>
#include <vector>

namespace {


#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            throw std::runtime_error("zeModuleCreate error"); \
        } \
    } while (false)

void *find_ze_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    HMODULE handle = LoadLibraryA("ze_loader.dll");
#endif
    if (!handle) {
        return nullptr;
    }

#if defined(__linux__)
    void *f = dlsym(handle, symbol);
#elif defined(_WIN32)
    void *f = GetProcAddress(handle, symbol);
#endif
    if (!f) {
        assert(!"not expected");
    }
    return f;
}

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)find_ze_symbol(symbol);
}

void func_zeModuleCreate(ze_context_handle_t hContext,
                         ze_device_handle_t hDevice, const ze_module_desc_t *desc,
                         ze_module_handle_t *phModule,
                         ze_module_build_log_handle_t *phBuildLog) {
    static auto f = find_ze_symbol<decltype(&zeModuleCreate)>("zeModuleCreate");

    if (!f)
        throw std::runtime_error("zeModuleCreate was not found");
    ZE_CHECK(f(hContext, hDevice, desc, phModule, phBuildLog));
}

std::unique_ptr<cl::sycl::program> sycl_create_program_with_level_zero(const cldnn::sycl::sycl_engine& engine, std::vector<uint8_t> binary) {
    auto desc = ze_module_desc_t();
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = binary.size();
    desc.pInputModule = binary.data();
    desc.pBuildFlags = "";
    desc.pConstants = nullptr;

    ze_module_handle_t ze_module;

    auto ze_device = engine.get_sycl_device().get_native<cl::sycl::backend::level_zero>();
    auto ze_ctx = engine.get_sycl_context().get_native<cl::sycl::backend::level_zero>();
    func_zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr);
    return std::unique_ptr<cl::sycl::program>(new cl::sycl::program(cl::sycl::level_zero::make<cl::sycl::program>(engine.get_sycl_context(), ze_module)));
}
} // namespace

namespace cldnn {
namespace sycl {

static std::vector<uint8_t> get_binary(cl_kernel kernel) {
    // Get the corresponding program object for the kernel
    cl_program program;
    cl_int error = clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_KERNEL_PROGRAM: " + std::to_string(error));
    }

    // Get the size of the program binary in bytes.
    size_t binary_size = 0;
    error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_PROGRAM_BINARY_SIZES: " + std::to_string(error));
    }

    // Binary is not available for the device.
    if (binary_size == 0)
        throw std::runtime_error("get_binary: Binary size is zero");

    // Get program binary.
    std::vector<uint8_t> binary(binary_size);
    uint8_t* binary_buffer = binary.data();
    error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_size, &binary_buffer, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_PROGRAM_BINARIES: " + std::to_string(error));
    }

    return binary;
}

std::shared_ptr<kernel> create_sycl_kernel(engine& engine, cl_context context, cl_kernel kernel, std::string entry_point) {
    if (engine.runtime_type() == runtime_types::l0) {
        std::cerr << "create kernel for l0 backend\n";
        auto binary = get_binary(kernel);
        std::unique_ptr<cl::sycl::program> prog = sycl_create_program_with_level_zero(dynamic_cast<sycl::sycl_engine&>(engine), binary);
        if (prog->get_state() == cl::sycl::program_state::none || prog->is_host()) {
            throw std::runtime_error("Invalid program state for sycl program");
        }
        cl::sycl::kernel k = prog->get_kernel(entry_point);
        return std::make_shared<sycl::sycl_kernel>(k, entry_point);
    } else {
        std::cerr << "create kernel for ocl backend\n";
        cl::sycl::kernel k(kernel, context);
        return std::make_shared<sycl::sycl_kernel>(k, entry_point);
    }
}

}  // namespace sycl
}  // namespace cldnn
