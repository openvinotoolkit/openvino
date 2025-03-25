// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/compiler.hpp"

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
    desc.add<DYNAMIC_SHAPE_TO_STATIC>();
    desc.add<EXECUTION_MODE_HINT>();
    desc.add<COMPILER_DYNAMIC_QUANTIZATION>();
    desc.add<QDQ_OPTIMIZATION>();
    desc.add<BATCH_COMPILER_MODE_SETTINGS>();
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
