// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler.hpp"

#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/itt.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu_private_properties.hpp"

#ifdef ENABLE_DRIVER_COMPILER_ADAPTER
#    include "driver_compiler_adapter.hpp"
#endif

#include <memory>
#include <string>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

using namespace intel_npu;

namespace {

std::shared_ptr<void> loadLibrary(const std::string& libpath) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    return ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
    return ov::util::load_shared_object(libpath.c_str());
#endif
}

std::shared_ptr<ICompiler> getCompiler(std::shared_ptr<void> so) {
    static constexpr auto CreateFuncName = "CreateNPUCompiler";
    auto symbol = ov::util::get_symbol(so, CreateFuncName);

    using CreateFuncT = void (*)(std::shared_ptr<ICompiler>&);
    const auto createFunc = reinterpret_cast<CreateFuncT>(symbol);

    std::shared_ptr<ICompiler> compilerPtr;
    createFunc(compilerPtr);
    return compilerPtr;
}

ov::SoPtr<ICompiler> loadCompiler(const std::string& libpath) {
    auto compilerSO = loadLibrary(libpath);
    auto compiler = getCompiler(compilerSO);

    return ov::SoPtr<ICompiler>(compiler, compilerSO);
}

ov::SoPtr<ICompiler> createCompilerAdapter(const Logger& log) {
    log.info("Driver compiler will be used.");
#ifdef ENABLE_DRIVER_COMPILER_ADAPTER
    const auto compilerInterface = std::make_shared<driverCompilerAdapter::LevelZeroCompilerAdapter>();
    return ov::SoPtr<ICompiler>(compilerInterface);
#else
    OPENVINO_THROW("NPU Compiler Adapter is not enabled");
#endif
}

ov::SoPtr<ICompiler> createNPUCompiler(const Logger& log) {
    log.info("MLIR compiler will be used.");
    std::string baseName = "npu_mlir_compiler";
    auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
    return loadCompiler(libPath);
}

ov::SoPtr<ICompiler> createCompilerImpl(ov::intel_npu::CompilerType compilerType, const Logger& log) {
    switch (compilerType) {
    case ov::intel_npu::CompilerType::MLIR:
        return createNPUCompiler(log);
    case ov::intel_npu::CompilerType::DRIVER:
        return createCompilerAdapter(log);
    default:
        OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
    }
}

}  // namespace

ov::SoPtr<ICompiler> intel_npu::createCompiler(ov::intel_npu::CompilerType compilerType, const Logger& log) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "intel_npu::createCompiler");
    auto logger = log.clone("createCompiler");
    try {
        return createCompilerImpl(compilerType, logger);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Got an error during compiler creation: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("Got an unknown error during compiler creation");
    }
}
