// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

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
#include "zero_link.hpp"

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

    auto zeroDevice = std::dynamic_pointer_cast<ZeroDevice>(zeroBackend->getDevice());
    if (zeroDevice == nullptr) {
        return;
    }

    ze_context_handle_t zeContext = static_cast<ze_context_handle_t>(zeroBackend->getContext());
    ze_driver_handle_t driverHandle = static_cast<ze_driver_handle_t>(zeroBackend->getDriverHandle());
    ze_device_handle_t deviceHandle = static_cast<ze_device_handle_t>(zeroBackend->getDeviceHandle());
    ze_graph_dditable_ext_curr_t& graphDdiTableExt = zeroBackend->getGraphDdiTable();
    ze_command_queue_npu_dditable_ext_curr_t& commandQueueDdiTable = zeroBackend->getCommandQueueDdiTable();

    uint32_t graphExtVersion = graphDdiTableExt.version();
    auto group_ordinal = zeroDevice->getGroupOrdinal();

    if (driverHandle == nullptr) {
        OPENVINO_THROW("PluginCompilerAdapter failed to get properties about Driver");
    }

    _logger.info("PluginCompilerAdapter creating adapter using graphExtVersion");

    switch (graphExtVersion) {
    case ZE_GRAPH_EXT_VERSION_1_3:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_3_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_4:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_4_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_5:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_5_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_6:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_6_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_7:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_7_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    case ZE_GRAPH_EXT_VERSION_1_8:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_8_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
        break;
    default:
        _zeroLink = std::make_shared<ZeroLink<ze_graph_dditable_ext_1_2_t>>(driverHandle,
                                                                            deviceHandle,
                                                                            zeContext,
                                                                            graphDdiTableExt,
                                                                            commandQueueDdiTable,
                                                                            group_ordinal);
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
    if (_zeroLink) {
        graphHandle = _zeroLink->getGraphHandle(networkDesc.compiledNetwork);
    }

    auto graph = std::make_shared<CipGraph>(_zeroLink,
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
    if (_zeroLink) {
        graphHandle = _zeroLink->getGraphHandle(network);
    }

    return std::make_shared<CipGraph>(_zeroLink, _compiler, graphHandle, std::move(networkMeta), network, config);
}

ov::SupportedOpsMap PluginCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "query");

    return _compiler->query(model, config);
}

}  // namespace intel_npu
