// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_clspv_compiler.hpp"

#include <clspv/Compiler.h>

#include <mutex>
#include <set>
#include <sstream>
#include <utility>

#include "openvino/core/except.hpp"
#include "vulkan_kernel_interface.hpp"

namespace cldnn::vulkan {
namespace {

std::mutex& get_clspv_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string translate_source_options(const std::string& source_options) {
    static const std::set<std::string> supported_flags{
        "-cl-fast-relaxed-math",
        "-cl-finite-math-only",
        "-cl-mad-enable",
        "-cl-no-signed-zeros",
        "-cl-no-subgroup-ifp",
        "-cl-unsafe-math-optimizations",
    };

    std::istringstream stream(source_options);
    std::string translated;
    for (std::string option; stream >> option;) {
        if (supported_flags.count(option) != 0 || option.rfind("-D", 0) == 0) {
            translated += " " + option;
        }
    }
    return translated;
}

}  // namespace

std::string vulkan_clspv_compiler::identity() {
    return OV_GPU_CLSPV_COMPILER_ID;
}

std::string vulkan_clspv_compiler::canonical_options(const std::string& source_options) {
    return "--spv-version=1.6 --cl-std=CL1.2 --inline-entry-points --pod-pushconstant "
           "--max-pushconstant-size=128 --long-vector" +
           translate_source_options(source_options);
}

vulkan_clspv_compilation vulkan_clspv_compiler::compile(const std::string& source, const std::string& source_options, const std::string& entry_point) const {
    OPENVINO_ASSERT(!source.empty(), "[GPU][Vulkan] CLSPV cannot compile an empty translation unit");
    OPENVINO_ASSERT(!entry_point.empty(), "[GPU][Vulkan] CLSPV requires an explicit entry point");

    const auto options = canonical_options(source_options);
    const char* sources[] = {source.data()};
    const size_t source_sizes[] = {source.size()};
    char* output_binary = nullptr;
    size_t output_binary_size = 0;
    char* output_log = nullptr;

    ClspvError result;
    {
        std::lock_guard<std::mutex> lock(get_clspv_mutex());
        result = clspvCompileFromSourcesString(1, source_sizes, sources, options.c_str(), &output_binary, &output_binary_size, &output_log);
    }

    std::string diagnostics = output_log == nullptr ? std::string{} : std::string(output_log);
    std::vector<uint8_t> spirv;
    if (output_binary != nullptr && output_binary_size != 0) {
        const auto* begin = reinterpret_cast<const uint8_t*>(output_binary);
        spirv.assign(begin, begin + output_binary_size);
    }
    clspvFreeOutputBuildObjs(output_binary, output_log);

    OPENVINO_ASSERT(result == CLSPV_SUCCESS,
                    "[GPU][Vulkan] CLSPV failed to compile entry point '",
                    entry_point,
                    "' with options '",
                    options,
                    "':\n",
                    diagnostics);
    OPENVINO_ASSERT(!spirv.empty(), "[GPU][Vulkan] CLSPV returned an empty SPIR-V module for entry point '", entry_point, "'");

    const auto interface = vulkan_kernel_interface::reflect(spirv, entry_point);
    interface.validate_canonical_compute_abi(entry_point);
    return {std::move(spirv), std::move(diagnostics)};
}

}  // namespace cldnn::vulkan
