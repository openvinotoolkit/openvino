// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/al/config/compiler.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// register
//

void intel_npu::registerCompilerOptions(OptionsDesc& desc) {
    desc.add<COMPILER_TYPE>();
    desc.add<COMPILATION_MODE>();
    desc.add<COMPILATION_MODE_PARAMS>();
    desc.add<BACKEND_COMPILATION_PARAMS>();
    desc.add<COMPILATION_NUM_THREADS>();
    desc.add<DPU_GROUPS>();
    desc.add<TILES>();
    desc.add<STEPPING>();
    desc.add<MAX_TILES>();
    desc.add<DMA_ENGINES>();
    desc.add<USE_ELF_COMPILER_BACKEND>();
    desc.add<DYNAMIC_SHAPE_TO_STATIC>();
    desc.add<SERIALIZE_MODE>();
}

//
// COMPILER_TYPE
//

std::string_view ov::intel_npu::stringifyEnum(ov::intel_npu::CompilerType val) {
    switch (val) {
    case ov::intel_npu::CompilerType::MLIR:
        return "MLIR";
    case ov::intel_npu::CompilerType::DRIVER:
        return "DRIVER";
    default:
        return "<UNKNOWN>";
    }
}

std::string_view intel_npu::COMPILER_TYPE::envVar() {
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    return "IE_NPU_COMPILER_TYPE";
#else
    return "";
#endif
}

ov::intel_npu::CompilerType intel_npu::COMPILER_TYPE::defaultValue() {
    return ov::intel_npu::CompilerType::DRIVER;
}

ov::intel_npu::CompilerType intel_npu::COMPILER_TYPE::parse(std::string_view val) {
    if (val == stringifyEnum(ov::intel_npu::CompilerType::MLIR)) {
        return ov::intel_npu::CompilerType::MLIR;
    } else if (val == stringifyEnum(ov::intel_npu::CompilerType::DRIVER)) {
        return ov::intel_npu::CompilerType::DRIVER;
    }

    OPENVINO_THROW("Value '", val, "' is not a valid COMPILER_TYPE option");
}

std::string intel_npu::COMPILER_TYPE::toString(const ov::intel_npu::CompilerType& val) {
    std::stringstream strStream;
    if (val == ov::intel_npu::CompilerType::MLIR) {
        strStream << "MLIR";
    } else if (val == ov::intel_npu::CompilerType::DRIVER) {
        strStream << "DRIVER";
    } else {
        OPENVINO_THROW("No valid string for current LOG_LEVEL option");
    }

    return strStream.str();
}

//
// USE_ELF_COMPILER_BACKEND
//

std::string_view ov::intel_npu::stringifyEnum(ov::intel_npu::ElfCompilerBackend val) {
    switch (val) {
    case ov::intel_npu::ElfCompilerBackend::AUTO:
        return "AUTO";
    case ov::intel_npu::ElfCompilerBackend::NO:
        return "NO";
    case ov::intel_npu::ElfCompilerBackend::YES:
        return "YES";
    default:
        return "<UNKNOWN>";
    }
}

ov::intel_npu::ElfCompilerBackend intel_npu::USE_ELF_COMPILER_BACKEND::parse(std::string_view val) {
    if (val == stringifyEnum(ov::intel_npu::ElfCompilerBackend::AUTO)) {
        return ov::intel_npu::ElfCompilerBackend::AUTO;
    } else if (val == stringifyEnum(ov::intel_npu::ElfCompilerBackend::NO)) {
        return ov::intel_npu::ElfCompilerBackend::NO;
    } else if (val == stringifyEnum(ov::intel_npu::ElfCompilerBackend::YES)) {
        return ov::intel_npu::ElfCompilerBackend::YES;
    }

    OPENVINO_THROW("Value '", val, "' is not a valid USE_ELF_COMPILER_BACKEND option");
}

std::string intel_npu::USE_ELF_COMPILER_BACKEND::toString(const ov::intel_npu::ElfCompilerBackend& val) {
    std::stringstream strStream;
    if (val == ov::intel_npu::ElfCompilerBackend::AUTO) {
        strStream << "AUTO";
    } else if (val == ov::intel_npu::ElfCompilerBackend::NO) {
        strStream << "NO";
    } else if (val == ov::intel_npu::ElfCompilerBackend::YES) {
        strStream << "YES";
    } else {
        OPENVINO_THROW("No valid string for current USE_ELF_COMPILER_BACKEND option");
    }

    return strStream.str();
}

//
// COMPILER_TYPE
//

std::string_view ov::intel_npu::stringifyEnum(ov::intel_npu::SerializeMode val) {
    switch (val) {
    case ov::intel_npu::SerializeMode::STREAM:
        return "STREAM";
    case ov::intel_npu::SerializeMode::FILE:
        return "FILE";
    case ov::intel_npu::SerializeMode::RAW:
        return "RAW";
    default:
        return "<UNKNOWN>";
    }
}

std::string_view intel_npu::SERIALIZE_MODE::envVar() {
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    return "IE_NPU_SERIALIZE_MODE";
#else
    return "";
#endif
}

ov::intel_npu::SerializeMode intel_npu::SERIALIZE_MODE::defaultValue() {
    return ov::intel_npu::SerializeMode::STREAM;
}

ov::intel_npu::SerializeMode intel_npu::SERIALIZE_MODE::parse(std::string_view val) {
    if (val == stringifyEnum(ov::intel_npu::SerializeMode::STREAM)) {
        return ov::intel_npu::SerializeMode::STREAM;
    } else if (val == stringifyEnum(ov::intel_npu::SerializeMode::FILE)) {
        return ov::intel_npu::SerializeMode::FILE;
    } else if (val == stringifyEnum(ov::intel_npu::SerializeMode::RAW)) {
        return ov::intel_npu::SerializeMode::RAW;
    }

    OPENVINO_THROW("Value '", val, "' is not a valid SERIALIZE_MODE option");
}

std::string intel_npu::SERIALIZE_MODE::toString(const ov::intel_npu::SerializeMode& val) {
    std::stringstream strStream;
    if (val == ov::intel_npu::SerializeMode::STREAM) {
        strStream << "STREAM";
    } else if (val == ov::intel_npu::SerializeMode::FILE) {
        strStream << "FILE";
    } else if (val == ov::intel_npu::SerializeMode::RAW) {
        strStream << "RAW";
    } else {
        OPENVINO_THROW("No valid string for current LOG_LEVEL option");
    }

    return strStream.str();
}