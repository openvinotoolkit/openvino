// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <ze_graph_ext.h>

#include <memory>
#include <string>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/icompiler.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "plugin_graph.hpp"
#include "zero_adapter.hpp"
#include "zero_backend.hpp"

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

PluginCompilerAdapter::PluginCompilerAdapter(const std::shared_ptr<IEngineBackend>& iEngineBackend)
    : _logger("PluginCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize PluginCompilerAdapter start");

    _logger.info("MLIR compiler will be used.");
    std::string baseName = "npu_mlir_compiler";
    auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
    _compiler = loadCompiler(libPath);

    if (iEngineBackend == nullptr) {
        return;
    }

    auto zeroBackend = std::dynamic_pointer_cast<ZeroEngineBackend>(iEngineBackend);
    if (zeroBackend == nullptr) {
        return;
    }

    ze_context_handle_t zeContext = static_cast<ze_context_handle_t>(zeroBackend->getContext());
    ze_device_handle_t deviceHandle = static_cast<ze_device_handle_t>(zeroBackend->getDeviceHandle());
    ze_graph_dditable_ext_curr_t& graphDdiTableExt = zeroBackend->getGraphDdiTable();
    ze_command_queue_npu_dditable_ext_curr_t& commandQueueDdiTable = zeroBackend->getCommandQueueDdiTable();

    uint32_t graphExtVersion = graphDdiTableExt.version();

    ze_device_properties_t deviceProperties = {};
    deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties", zeDeviceGetProperties(deviceHandle, &deviceProperties));
    auto groupOrdinal = zeroUtils::findGroupOrdinal(deviceHandle, deviceProperties);

    _logger.info("PluginCompilerAdapter creating adapter using graphExtVersion");

    switch (graphExtVersion) {
    case ZE_GRAPH_EXT_VERSION_1_3:
        _zeroAdapter = std::make_shared<ZeroAdapter<ze_graph_dditable_ext_1_3_t>>(deviceHandle,
                                                                                  zeContext,
                                                                                  graphDdiTableExt,
                                                                                  commandQueueDdiTable,
                                                                                  groupOrdinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_4:
        _zeroAdapter = std::make_shared<ZeroAdapter<ze_graph_dditable_ext_1_4_t>>(deviceHandle,
                                                                                  zeContext,
                                                                                  graphDdiTableExt,
                                                                                  commandQueueDdiTable,
                                                                                  groupOrdinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_5:
        _zeroAdapter = std::make_shared<ZeroAdapter<ze_graph_dditable_ext_1_5_t>>(deviceHandle,
                                                                                  zeContext,
                                                                                  graphDdiTableExt,
                                                                                  commandQueueDdiTable,
                                                                                  groupOrdinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_6:
        _zeroAdapter = std::make_shared<ZeroAdapter<ze_graph_dditable_ext_1_6_t>>(deviceHandle,
                                                                                  zeContext,
                                                                                  graphDdiTableExt,
                                                                                  commandQueueDdiTable,
                                                                                  groupOrdinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_7:
        _zeroAdapter = std::make_shared<ZeroAdapter<ze_graph_dditable_ext_1_7_t>>(deviceHandle,
                                                                                  zeContext,
                                                                                  graphDdiTableExt,
                                                                                  commandQueueDdiTable,
                                                                                  groupOrdinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_8:
        _zeroAdapter = std::make_shared<ZeroAdapter<ze_graph_dditable_ext_1_8_t>>(deviceHandle,
                                                                                  zeContext,
                                                                                  graphDdiTableExt,
                                                                                  commandQueueDdiTable,
                                                                                  groupOrdinal);
        break;
    default:
        _zeroAdapter = std::make_shared<ZeroAdapter<ze_graph_dditable_ext_1_2_t>>(deviceHandle,
                                                                                  zeContext,
                                                                                  graphDdiTableExt,
                                                                                  commandQueueDdiTable,
                                                                                  groupOrdinal);
        break;
    }

    _logger.info("initialize PluginCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compile");

    _logger.debug("compile start");
    auto networkDesc = _compiler->compile(model, config);
    _logger.debug("compile end");

    ze_graph_handle_t graphHandle = nullptr;
    if (_zeroAdapter) {
        graphHandle = _zeroAdapter->getGraphHandle(networkDesc.compiledNetwork);
    }

    auto graph = std::make_shared<PluginGraph>(_zeroAdapter,
                                               _compiler,
                                               graphHandle,
                                               std::move(networkDesc.metadata),
                                               std::move(networkDesc.compiledNetwork),
                                               config);

    return graph;
}

std::shared_ptr<IGraph> PluginCompilerAdapter::parse(const std::vector<uint8_t>& network, const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "parse");

    _logger.debug("parse start");
    auto networkMeta = _compiler->parse(network, config);
    _logger.debug("parse end");

    ze_graph_handle_t graphHandle = nullptr;
    if (_zeroAdapter) {
        graphHandle = _zeroAdapter->getGraphHandle(network);
    }

    return std::make_shared<PluginGraph>(_zeroAdapter, _compiler, graphHandle, std::move(networkMeta), network, config);
}

ov::SupportedOpsMap PluginCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "query");

    return _compiler->query(model, config);
}

}  // namespace intel_npu
