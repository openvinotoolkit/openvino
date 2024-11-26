// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <memory>
#include <string>

#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "plugin_graph.hpp"

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

PluginCompilerAdapter::PluginCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("PluginCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize PluginCompilerAdapter start");

    _logger.info("MLIR compiler will be used.");
    std::string baseName = "npu_mlir_compiler";
    auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
    _compiler = loadCompiler(libPath);

    if (_zeroInitStruct == nullptr) {
        return;
    }

    uint32_t graphExtVersion = _zeroInitStruct->getGraphDdiTable().version();

    _logger.info("PluginCompilerAdapter creating adapter using graphExtVersion");

    _zeGraphExt = std::make_shared<ZeGraphExtWrappers>(_zeroInitStruct);

    _logger.info("initialize PluginCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compile");

    _logger.debug("compile start");
    auto networkDesc = _compiler->compile(model, config);
    auto networkSO = std::make_shared<std::vector<uint8_t>>(std::move(networkDesc.compiledNetwork));
    _logger.debug("compile end");

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to get the graph handle from the compiled network
        try {
            graphHandle = _zeGraphExt->getGraphHandle(networkSO->data(), networkSO->size());
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }
    auto networkSOPtr = std::make_shared<ov::SharedBuffer<std::shared_ptr<std::vector<uint8_t>>>>(reinterpret_cast<char*>(networkSO->data()), networkSO->size(), networkSO);
    return std::make_shared<PluginGraph>(_zeGraphExt,
                                         _compiler,
                                         _zeroInitStruct,
                                         graphHandle,
                                         std::move(networkDesc.metadata),
                                         networkSOPtr,
                                         config);
}

std::shared_ptr<IGraph> PluginCompilerAdapter::parse(std::shared_ptr<ov::AlignedBuffer> networkSOPtr, const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "parse");

    _logger.debug("parse start");
    std::vector<uint8_t> network(networkSOPtr->size());
    network.assign(reinterpret_cast<uint8_t*>(networkSOPtr->get_ptr()), reinterpret_cast<uint8_t*>(networkSOPtr->get_ptr()) + networkSOPtr->size());
    auto networkMeta = _compiler->parse(network, config);
    network.clear();
    network.shrink_to_fit();
    _logger.debug("parse end");

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        graphHandle = _zeGraphExt->getGraphHandle(reinterpret_cast<const uint8_t*>(networkSOPtr->get_ptr()), networkSOPtr->size());
    }

    return std::make_shared<PluginGraph>(_zeGraphExt,
                                         _compiler,
                                         _zeroInitStruct,
                                         graphHandle,
                                         std::move(networkMeta),
                                         std::move(networkSOPtr),
                                         config);
}

ov::SupportedOpsMap PluginCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "query");

    return _compiler->query(model, config);
}

uint32_t PluginCompilerAdapter::get_version() const {
    return _compiler->get_version();
}

}  // namespace intel_npu
