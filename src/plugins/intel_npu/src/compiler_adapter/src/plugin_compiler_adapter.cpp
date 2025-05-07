// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <memory>
#include <string>

#include "graph.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

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
    _logger.debug("compile end");

    auto tensor =
        ov::Tensor(ov::element::u8, ov::Shape{networkDesc.compiledNetwork.size()}, networkDesc.compiledNetwork.data());
    auto impl = ov::get_tensor_impl(tensor);
    std::shared_ptr<std::vector<uint8_t>> sharedCompiledNetwork =
        std::make_shared<std::vector<uint8_t>>(std::move(networkDesc.compiledNetwork));
    impl._so = std::move(sharedCompiledNetwork);
    tensor = ov::make_tensor(impl);

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to get the graph handle from the compiled
        // network
        try {
            graphHandle =
                _zeGraphExt->getGraphHandle(*reinterpret_cast<const uint8_t*>(tensor.data()), tensor.get_byte_size());
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }

    return std::make_shared<Graph>(_zeGraphExt,
                                   _zeroInitStruct,
                                   graphHandle,
                                   std::move(networkDesc.metadata),
                                   std::move(tensor),
                                   /* blobAllocatedByPlugin = */ true,
                                   config,
                                   _compiler);
}

std::shared_ptr<IGraph> PluginCompilerAdapter::parse(ov::Tensor blob,
                                                     bool blobAllocatedByPlugin,
                                                     const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "parse");

    _logger.debug("parse start");
    std::vector<uint8_t> network(blob.get_byte_size());
    network.assign(reinterpret_cast<const uint8_t*>(blob.data()),
                   reinterpret_cast<const uint8_t*>(blob.data()) + blob.get_byte_size());
    auto networkMeta = _compiler->parse(network, config);
    network.clear();
    network.shrink_to_fit();
    _logger.debug("parse end");

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        graphHandle = _zeGraphExt->getGraphHandle(*reinterpret_cast<const uint8_t*>(blob.data()), blob.get_byte_size());
    }

    return std::make_shared<Graph>(_zeGraphExt,
                                   _zeroInitStruct,
                                   graphHandle,
                                   std::move(networkMeta),
                                   std::move(blob),
                                   blobAllocatedByPlugin,
                                   config,
                                   _compiler);
}

ov::SupportedOpsMap PluginCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "query");

    return _compiler->query(model, config);
}

uint32_t PluginCompilerAdapter::get_version() const {
    // returning max val as PluginCompiler supports all features and options the plugin is aware of
    return _compiler->get_version();
}

std::vector<std::string> PluginCompilerAdapter::get_supported_options() const {
    // PluginCompiler has all the same options as plugin
    // Returing empty string to let the plugin fallback to legacy registration
    return {};
}

bool PluginCompilerAdapter::is_option_supported(std::string optname) const {
    // This functions has no utility in PluginCompiler
    // returning false for any request to avoid the option of spaming the plugin
    return false;
}

}  // namespace intel_npu
