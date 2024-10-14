// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cip_compiler_adapter.hpp"

#include <ze_graph_ext.h>

#include <memory>
#include <string>

#include "cip_graph.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/al/itt.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "npu_private_properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "ze_intel_npu_uuid.h"
#include "zero_backend.hpp"
#include "zero_device.hpp"
#include "zero_init.hpp"

namespace {
std::shared_ptr<void> loadLibrary(const std::string& libpath) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    return ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
    return ov::util::load_shared_object(libpath.c_str());
#endif
}

std::shared_ptr<intel_npu::ICompiler> getCompiler(std::shared_ptr<void> so) {
    static constexpr auto CreateFuncName = "CreateNPUCompiler";
    auto symbol = ov::util::get_symbol(so, CreateFuncName);

    using CreateFuncT = void (*)(std::shared_ptr<intel_npu::ICompiler>&);
    const auto createFunc = reinterpret_cast<CreateFuncT>(symbol);

    std::shared_ptr<intel_npu::ICompiler> compilerPtr;
    createFunc(compilerPtr);
    return compilerPtr;
}

ov::SoPtr<intel_npu::ICompiler> loadCompiler(const std::string& libpath) {
    auto compilerSO = loadLibrary(libpath);
    auto compiler = getCompiler(compilerSO);

    return ov::SoPtr<intel_npu::ICompiler>(compiler, compilerSO);
}
}  // namespace

namespace intel_npu {

CipCompilerAdapter::CipCompilerAdapter(const std::shared_ptr<IEngineBackend>& iEngineBackend)
    : _iEngineBackend(iEngineBackend),
      _logger("CipCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize CipCompilerAdapter start");

    _logger.info("MLIR compiler will be used.");
    std::string baseName = "npu_mlir_compiler";
    auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
    _compiler = loadCompiler(libPath);
}

std::shared_ptr<IGraph> CipCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                    const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "CipCompilerAdapter", "compile");

    _logger.debug("compile start");
    auto networkDesc = _compiler->compile(model, config);
    _logger.debug("compile end");

    return std::make_shared<CipGraph>(_iEngineBackend,
                                      std::move(networkDesc.metadata),
                                      std::move(networkDesc.compiledNetwork),
                                      config);
}

std::shared_ptr<IGraph> CipCompilerAdapter::parse(const std::vector<uint8_t>& network, const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "CipCompilerAdapter", "parse");

    _logger.debug("parse start");
    auto networkMeta = _compiler->parse(network, config);
    _logger.debug("parse end");

    return std::make_shared<CipGraph>(_iEngineBackend, std::move(networkMeta), std::move(network), config);
}

ov::SupportedOpsMap CipCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                              const Config& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "CipCompilerAdapter", "query");

    return _compiler->query(model, config);
}

}  // namespace intel_npu
