// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

#include "common.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace ov {

namespace intel_npu {

std::string_view stringifyEnum(CompilerType val);
std::string_view stringifyEnum(ElfCompilerBackend val);

}  // namespace intel_npu

}  // namespace ov

namespace intel_npu {

//
// register
//

void registerCompilerOptions(OptionsDesc& desc);

//
// COMPILER_TYPE
//

struct COMPILER_TYPE final : OptionBase<COMPILER_TYPE, ov::intel_npu::CompilerType> {
    static std::string_view key() {
        return ov::intel_npu::compiler_type.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::intel_npu::CompilerType";
    }

    static std::string_view envVar();

    static ov::intel_npu::CompilerType defaultValue();

    static ov::intel_npu::CompilerType parse(std::string_view val);

    static std::string toString(const ov::intel_npu::CompilerType& val);

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// COMPILATION_MODE
//

struct COMPILATION_MODE final : OptionBase<COMPILATION_MODE, std::string> {
    static std::string_view key() {
        return ov::intel_npu::compilation_mode.name();
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_COMPILATION_MODE";
    }
#endif

    static std::string defaultValue() {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// EXECUTION_MODE_HINT
//

struct EXECUTION_MODE_HINT final : OptionBase<EXECUTION_MODE_HINT, ov::hint::ExecutionMode> {
    static std::string_view key() {
        return ov::hint::execution_mode.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::hint::ExecutionMode";
    }

    static ov::hint::ExecutionMode defaultValue() {
        return ov::hint::ExecutionMode::PERFORMANCE;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

//
// DYNAMIC_SHAPE_TO_STATIC
//

struct DYNAMIC_SHAPE_TO_STATIC final : OptionBase<DYNAMIC_SHAPE_TO_STATIC, bool> {
    static std::string_view key() {
        return ov::intel_npu::dynamic_shape_to_static.name();
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DYNAMIC_SHAPE_TO_STATIC";
    }
#endif

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

//
// COMPILATION_MODE_PARAMS
//

struct COMPILATION_MODE_PARAMS final : OptionBase<COMPILATION_MODE_PARAMS, std::string> {
    static std::string_view key() {
        return ov::intel_npu::compilation_mode_params.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};

//
// DPU_GROUPS
//

struct DPU_GROUPS final : OptionBase<DPU_GROUPS, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::dpu_groups.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DPU_GROUPS";
    }
#endif
};

//
// SELECTED_TILES
//

struct TILES final : OptionBase<TILES, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::tiles.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_TILES";
    }
#endif
};

//
// STEPPING
//

struct STEPPING final : OptionBase<STEPPING, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::stepping.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// MAX_TILES
//

struct MAX_TILES final : OptionBase<MAX_TILES, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::max_tiles.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// DMA_ENGINES
//

struct DMA_ENGINES final : OptionBase<DMA_ENGINES, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::dma_engines.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DMA_ENGINES";
    }
#endif
};

//
// USE_ELF_COMPILER_BACKEND
//

struct USE_ELF_COMPILER_BACKEND final : OptionBase<USE_ELF_COMPILER_BACKEND, ov::intel_npu::ElfCompilerBackend> {
    static std::string_view key() {
        return ov::intel_npu::use_elf_compiler_backend.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::intel_npu::ElfCompilerBackend";
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_USE_ELF_COMPILER_BACKEND";
    }
#endif

    static ov::intel_npu::ElfCompilerBackend defaultValue() {
        return ov::intel_npu::ElfCompilerBackend::AUTO;
    }

    static ov::intel_npu::ElfCompilerBackend parse(std::string_view val);

    static std::string toString(const ov::intel_npu::ElfCompilerBackend& val);
};

//
// BACKEND_COMPILATION_PARAMS
//

struct BACKEND_COMPILATION_PARAMS final : OptionBase<BACKEND_COMPILATION_PARAMS, std::string> {
    static std::string_view key() {
        return ov::intel_npu::backend_compilation_params.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

//
// COMPILATION_NUM_THREADS
//

struct COMPILATION_NUM_THREADS final : OptionBase<COMPILATION_NUM_THREADS, int32_t> {
    static std::string_view key() {
        return ov::compilation_num_threads.name();
    }

    static int32_t defaultValue() {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }

    static void validateValue(const int32_t& num) {
        if (num <= 0) {
            OPENVINO_THROW("ov::compilation_num_threads must be positive int32 value");
        }
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

}  // namespace intel_npu
