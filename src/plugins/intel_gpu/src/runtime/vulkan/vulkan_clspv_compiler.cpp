// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_clspv_compiler.hpp"

#include <clspv/Compiler.h>
#include <spirv-tools/libspirv.hpp>

#include <cstring>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>

#include "openvino/core/except.hpp"
#include "vulkan_device.hpp"
#include "vulkan_kernel_interface.hpp"

namespace cldnn::vulkan {
namespace {

// Clang advertises this extension, but CLSPV does not implement its Intel builtins.
// Select the existing generic conversions in the kernel selector's bf16_utils.cl.
constexpr char type_preamble[] = "#undef cl_intel_bfloat16_conversions\n";

std::mutex& get_clspv_mutex() {
    static std::mutex mutex;
    return mutex;
}

void validate_spirv(const std::vector<uint8_t>& spirv, const std::string& entry_point) {
    OPENVINO_ASSERT(spirv.size() % sizeof(uint32_t) == 0, "[GPU][Vulkan] Invalid SPIR-V word alignment");
    std::vector<uint32_t> words(spirv.size() / sizeof(uint32_t));
    std::memcpy(words.data(), spirv.data(), spirv.size());
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
    std::string diagnostics;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        diagnostics += "instruction " + std::to_string(position.index) + ": " + message + "\n";
    });
    OPENVINO_ASSERT(tools.Validate(words),
                    "[GPU][Vulkan] CLSPV produced invalid Vulkan 1.3 SPIR-V for entry point '",
                    entry_point,
                    "':\n",
                    diagnostics);
}

void normalize_storage_capabilities(std::vector<uint8_t>& spirv, const vulkan_device& device) {
    if (device.supports_arithmetic_type(data_types::i8) && device.supports_arithmetic_type(data_types::i16) &&
        device.supports_arithmetic_type(data_types::f16)) {
        return;
    }
    OPENVINO_ASSERT(spirv.size() % sizeof(uint32_t) == 0, "[GPU][Vulkan] Invalid SPIR-V word alignment");
    std::vector<uint32_t> words(spirv.size() / sizeof(uint32_t));
    std::memcpy(words.data(), spirv.data(), spirv.size());
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
    std::string diagnostics;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        diagnostics += std::string(message) + "\n";
    });
    std::string assembly;
    OPENVINO_ASSERT(tools.Disassemble(words, &assembly, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER),
                    "[GPU][Vulkan] Cannot inspect CLSPV storage requirements: ", diagnostics);
    struct storage_capability {
        data_types type;
        const char* arithmetic;
        const char* storage;
    };
    bool changed = false;
    for (const auto& capability : {storage_capability{data_types::i8, "Int8", "StorageBuffer8BitAccess"},
                                   storage_capability{data_types::i16, "Int16", "StorageBuffer16BitAccess"},
                                   storage_capability{data_types::f16, "Float16", "StorageBuffer16BitAccess"}}) {
        const auto declaration = std::string("OpCapability ") + capability.arithmetic + "\n";
        const auto position = assembly.find(declaration);
        if (device.supports_arithmetic_type(capability.type) || position == std::string::npos) {
            continue;
        }
        OPENVINO_ASSERT(capability.type == data_types::i8 || device.supports_16bit_storage(),
                        "[GPU][Vulkan] Device lacks required 16-bit storage");
        const auto storage = std::string("OpCapability ") + capability.storage + "\n";
        assembly.replace(position, declaration.size(), assembly.find(storage) == std::string::npos ? storage : "");
        changed = true;
    }
    if (!changed) {
        return;
    }
    // CLSPV declares arithmetic capabilities whenever a narrow type occurs.
    // Only accept storage-only declarations if the unchanged instructions meet
    // the upstream Vulkan validator's complete contract. Never strip and submit.
    OPENVINO_ASSERT(tools.Assemble(assembly, &words, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS) && tools.Validate(words),
                    "[GPU][Vulkan] Kernel still requires unsupported narrow arithmetic: ", diagnostics);
    spirv.resize(words.size() * sizeof(uint32_t));
    std::memcpy(spirv.data(), words.data(), spirv.size());
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
    return OV_GPU_CLSPV_COMPILER_ID + std::string(type_preamble) + "storage-types-v1";
}

std::string vulkan_clspv_compiler::canonical_options(const std::string& source_options) {
    // Use Vulkan FMA rather than the builtin library's wide-integer emulation.
    // Keep the library implementations of other math functions, including pow.
    return "--spv-version=1.6 --cl-std=CL1.2 --inline-entry-points --pod-pushconstant "
           "--max-pushconstant-size=128 --long-vector --use-native-builtins=fma" +
           translate_source_options(source_options);
}

vulkan_clspv_compilation vulkan_clspv_compiler::compile(const std::string& source, const std::string& source_options, const std::string& entry_point,
                                                     const vulkan_device& device) const {
    OPENVINO_ASSERT(!source.empty(), "[GPU][Vulkan] CLSPV cannot compile an empty translation unit");
    OPENVINO_ASSERT(!entry_point.empty(), "[GPU][Vulkan] CLSPV requires an explicit entry point");

    const auto options = canonical_options(source_options);
    const auto compiler_source = std::string(type_preamble) + source;
    const char* sources[] = {compiler_source.data()};
    const size_t source_sizes[] = {compiler_source.size()};
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

    normalize_storage_capabilities(spirv, device);
    // Reflection checks the buffer ABI; it is not an instruction validator.
    // Validate native-arithmetic modules as well as storage-normalized output.
    validate_spirv(spirv, entry_point);
    const auto interface = vulkan_kernel_interface::reflect(spirv, entry_point);
    interface.validate_canonical_compute_abi(entry_point);
    return {std::move(spirv), std::move(diagnostics)};
}

}  // namespace cldnn::vulkan
