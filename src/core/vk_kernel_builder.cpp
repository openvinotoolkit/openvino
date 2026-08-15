// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_kernel_builder.hpp"
#include "vk_spirv_reflection.hpp"

// spirv_kernels.inc is generated at build time from kernels/*.comp
// (see cmake/gen_spirv.cmake). It is included at file scope because it
// contains anonymous-namespace declarations.
#include "spirv_kernels.inc"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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

namespace ov::core::vulkan {
namespace cross_platform {

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

vk_kernel_builder::vk_kernel_builder(VkDevice device, const vk_platform_config& config)
    : _device(device)
    , _config(config) {
    // Vulkan SC offline mode: all pipelines are created against this cache so
    // the fully-built pipeline state can be flushed to disk (see destructor).
    if (!_config.offline_pipeline_dir.empty()) {
        VkPipelineCacheCreateInfo cache_info{};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        VK_CALL(vkCreatePipelineCache(_device, &cache_info, nullptr, &_pipeline_cache), "vkCreatePipelineCache");
    }
}

vk_kernel_builder::~vk_kernel_builder() {
    // Offline artifacts: the pipeline cache blob plus per-kernel SPIR-V
    // (written in build_from_spirv). A Vulkan SC system consumes these instead
    // of creating pipelines at runtime.
    if (_pipeline_cache != VK_NULL_HANDLE) {
        size_t data_size = 0;
        const VkResult result = vkGetPipelineCacheData(_device, _pipeline_cache, &data_size, nullptr);
        if (result == VK_SUCCESS && data_size > 0) {
            std::vector<uint8_t> data(data_size);
            if (vkGetPipelineCacheData(_device, _pipeline_cache, &data_size, data.data()) == VK_SUCCESS) {
                std::error_code ec;
                const auto dir = std::filesystem::path(_config.offline_pipeline_dir);
                std::filesystem::create_directories(dir, ec);
                std::ofstream out(dir / "vk_pipeline_cache.bin", std::ios::binary);
                out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            }
        }
        vkDestroyPipelineCache(_device, _pipeline_cache, nullptr);
    }
}

void vk_kernel_builder::build_kernels(const void* src,
                                      size_t src_bytes,
                                      KernelFormat src_format,
                                      const std::string& options,
                                      std::vector<vk_kernel_ptr>& out) const {
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

void vk_kernel_builder::build_native_kernel(const std::string& id, std::vector<vk_kernel_ptr>& out) const {
    for (const auto& entry : native_kernels_table) {
        if (id == entry.id) {
            build_from_spirv(entry.spirv, entry.words * sizeof(uint32_t), out, entry.id);
            return;
        }
    }
    OPENVINO_THROW("[GPU] Vulkan kernel builder: no native kernel with id '", id, "' in the builtin store");
}

void vk_kernel_builder::build_from_spirv(const void* src,
                                         size_t src_bytes,
                                         std::vector<vk_kernel_ptr>& out,
                                         std::string_view artifact_base) const {
    if (src == nullptr || src_bytes == 0 || src_bytes % sizeof(uint32_t) != 0) {
        OPENVINO_THROW("[GPU] Vulkan kernel builder: invalid SPIR-V binary (", src_bytes, " bytes)");
    }
    const auto* words = reinterpret_cast<const uint32_t*>(src);
    std::vector<uint32_t> spirv(words, words + src_bytes / sizeof(uint32_t));

    const auto reflections = parse_spirv_reflection(spirv);
    if (reflections.empty()) {
        OPENVINO_THROW("[GPU] Vulkan kernel builder: no kernels found in SPIR-V module");
    }
    // Vulkan SC offline mode: dump the exact SPIR-V module — the artifact an
    // offline pipeline compiler consumes. Native kernels are named by their
    // store id; clspv source kernels fall back to the entry-point name.
    if (!_config.offline_pipeline_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(_config.offline_pipeline_dir), ec);
        const std::string file_stem = artifact_base.empty() ? reflections.front().name : std::string(artifact_base);
        const auto path = std::filesystem::path(_config.offline_pipeline_dir) / (file_stem + ".spv");
        std::ofstream out_file(path, std::ios::binary);
        out_file.write(reinterpret_cast<const char*>(spirv.data()),
                       static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
    }
    for (const auto& refl : reflections) {
        if (refl.name.empty())
            continue;
        out.push_back(vk_kernel::create_kernel(_device, spirv, refl, _pipeline_cache));
    }
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
