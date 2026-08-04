// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_kernel_builder.hpp"
#include "vk_kernel.hpp"
#include "vk_spirv_reflection.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef OV_GPU_WITH_CLSPV
// clspv is compiled into the plugin as a static library. The C API declared
// in clspv/include/clspv/Compiler.h is used to avoid a hard dependency on
// clspv headers at plugin build time. clspvFreeOutputBuildObjs from that
// header is static inline, so the returned buffers are released with free().
extern "C" {
typedef enum ClspvError {
    CLSPV_SUCCESS = 0,
    CLSPV_OUT_OF_HOST_MEM,
    CLSPV_INVALID_ARG,
    CLSPV_ERROR
} ClspvError;

ClspvError clspvCompileFromSourcesString(const size_t program_count,
                                         const size_t* program_sizes,
                                         const char** programs,
                                         const char* options,
                                         char** output_binary,
                                         size_t* output_binary_size,
                                         char** output_log);
}
#endif

namespace cldnn {
namespace vk {

namespace {

#ifdef OV_GPU_WITH_CLSPV
// Mirrors the SOURCE convention of the OpenCL builder: |src| is a pointer to
// a null-terminated C string (single program).
std::string source_to_string(const void* src, size_t src_bytes) {
    const char* str = *reinterpret_cast<const char* const*>(&src);
    if (str == nullptr)
        return {};
    if (src_bytes != 0)
        return std::string(str, src_bytes);
    return std::string(str);
}

std::vector<uint32_t> compile_with_clspv(const std::string& source, const std::string& options) {
    std::vector<uint32_t> spirv;
    if (source.empty())
        return spirv;

    const char* programs[] = {source.c_str()};
    size_t program_sizes[] = {source.size()};

    char* binary = nullptr;
    size_t binary_size = 0;
    char* log = nullptr;
    ClspvError err = clspvCompileFromSourcesString(1, program_sizes, programs, options.c_str(), &binary, &binary_size, &log);
    if (err != CLSPV_SUCCESS) {
        std::string log_str = log != nullptr ? std::string(log) : std::string();
        std::free(binary);
        std::free(log);
        OPENVINO_THROW("[GPU] clspv failed to compile kernel (error ", static_cast<int>(err), "): ", log_str);
    }
    if (binary == nullptr || binary_size % sizeof(uint32_t) != 0) {
        std::free(binary);
        std::free(log);
        OPENVINO_THROW("[GPU] clspv produced invalid SPIR-V output (", binary_size, " bytes)");
    }

    spirv.resize(binary_size / sizeof(uint32_t));
    std::memcpy(spirv.data(), binary, binary_size);
    std::free(binary);
    std::free(log);
    return spirv;
}
#endif  // OV_GPU_WITH_CLSPV

}  // namespace

void vk_kernel_builder::build_kernels(const void* src,
                                      size_t src_bytes,
                                      KernelFormat src_format,
                                      const std::string& options,
                                      std::vector<kernel::ptr>& out) const {
    switch (src_format) {
    case KernelFormat::SOURCE: {
#ifdef OV_GPU_WITH_CLSPV
        const std::string source = source_to_string(src, src_bytes);
        const auto spirv = compile_with_clspv(source, options);
        build_from_spirv(spirv.data(), spirv.size() * sizeof(uint32_t), out);
#else
        OPENVINO_THROW("[GPU] Vulkan kernel builder: clspv support is not enabled, rebuild with OV_GPU_VULKAN_CLSPV_PATH");
#endif
        break;
    }
    case KernelFormat::NATIVE_BIN: {
        build_from_spirv(src, src_bytes, out);
        break;
    }
    default:
        OPENVINO_THROW("[GPU] Trying to build kernel from unexpected format");
        break;
    }
}

void vk_kernel_builder::build_from_spirv(const void* src, size_t src_bytes, std::vector<kernel::ptr>& out) const {
    if (src == nullptr || src_bytes == 0 || src_bytes % sizeof(uint32_t) != 0) {
        OPENVINO_THROW("[GPU] Vulkan kernel builder: invalid SPIR-V binary (", src_bytes, " bytes)");
    }
    const auto* words = reinterpret_cast<const uint32_t*>(src);
    std::vector<uint32_t> spirv(words, words + src_bytes / sizeof(uint32_t));

    const auto reflections = parse_spirv_reflection(spirv);
    if (reflections.empty()) {
        OPENVINO_THROW("[GPU] Vulkan kernel builder: no kernels found in SPIR-V module");
    }
    for (const auto& refl : reflections) {
        if (refl.name.empty())
            continue;
        out.push_back(vk_kernel::create_kernel(_device, spirv, refl));
    }
}

}  // namespace vk
}  // namespace cldnn
