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
#include "mem_usage.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "weightless_graph.hpp"

namespace {

std::shared_ptr<void> load_library(const std::string& libpath) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    return ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
    return ov::util::load_shared_object(libpath.c_str());
#endif
}

std::shared_ptr<intel_npu::ICompiler> get_compiler(std::shared_ptr<void> so) {
    static constexpr auto CreateFuncName = "CreateNPUCompiler";
    auto symbol = ov::util::get_symbol(so, CreateFuncName);

    using CreateFuncT = void (*)(std::shared_ptr<intel_npu::ICompiler>&);
    const auto createFunc = reinterpret_cast<CreateFuncT>(symbol);

    std::shared_ptr<intel_npu::ICompiler> compilerPtr;
    createFunc(compilerPtr);
    return compilerPtr;
}

ov::SoPtr<intel_npu::ICompiler> load_compiler(const std::string& libpath) {
    auto compilerSO = load_library(libpath);
    auto compiler = get_compiler(compilerSO);

    return ov::SoPtr<intel_npu::ICompiler>(compiler, compilerSO);
}

ov::Tensor make_tensor_from_vector(std::vector<uint8_t>& vector) {
    auto tensor = ov::Tensor(ov::element::u8, ov::Shape{vector.size()}, vector.data());
    auto impl = ov::get_tensor_impl(std::move(tensor));
    std::shared_ptr<std::vector<uint8_t>> sharedCompiledNetwork =
        std::make_shared<std::vector<uint8_t>>(std::move(vector));
    impl._so = std::move(sharedCompiledNetwork);
    return ov::make_tensor(impl);
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
    _compiler = load_compiler(libPath);

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

    ov::Tensor tensor = make_tensor_from_vector(networkDesc.compiledNetwork);
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
                                   /* persistentBlob = */ true,
                                   config,
                                   _compiler);
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compileWS(const std::shared_ptr<ov::Model>& model,
                                                         const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compileWS");

    std::vector<std::shared_ptr<NetworkDescription>> initNetworkDescriptions;
    std::shared_ptr<NetworkDescription> mainNetworkDescription;

    _logger.debug("compile start");

    const auto starts_with = [](const std::string& str, const std::string& prefix) {
        return str.substr(0, prefix.size()) == prefix;
    };
    const auto isInit = [&](std::string name) {
        return starts_with(name, "init");
    };

    const auto isMain = [&](std::string name) {
        return starts_with(name, "main");
    };

    _logger.info("SEPARATE_WEIGHTS_VERSION: ", config.get<SEPARATE_WEIGHTS_VERSION>());

    int64_t compile_model_mem_start = 0;
    if (_logger.level() >= ov::log::Level::INFO) {
        compile_model_mem_start = get_peak_memory_usage();
    }
    switch (config.get<SEPARATE_WEIGHTS_VERSION>()) {
    case ov::intel_npu::WSVersion::ONE_SHOT: {
        std::vector<std::shared_ptr<NetworkDescription>> initMainNetworkDescriptions =
            _compiler->compileWsOneShot(model, config);

#if 0  // TODO: it is not clear whether we should change the name
            OPENVINO_ASSERT(isMain(initMainNetworkDescriptions.back()->metadata.name),
                            "Unexpected network name for main:",
                            initMainNetworkDescriptions.back()->metadata.name);
#endif

        mainNetworkDescription = initMainNetworkDescriptions.back();
        initMainNetworkDescriptions.pop_back();
        initNetworkDescriptions = std::move(initMainNetworkDescriptions);
    } break;
    case ov::intel_npu::WSVersion::ITERATIVE: {
        const std::shared_ptr<ov::Model> originalModel = model->clone();
        std::shared_ptr<ov::Model> targetModel = model;
        size_t i = 0;

        while (auto networkDescription =
                   std::make_shared<NetworkDescription>(_compiler->compileWsIterative(targetModel, config, i++))) {
            if (isInit(networkDescription->metadata.name)) {
                initNetworkDescriptions.push_back(networkDescription);
                targetModel = originalModel->clone();
                continue;
            }
            OPENVINO_ASSERT(isMain(networkDescription->metadata.name),
                            "Unexpected network name: ",
                            networkDescription->metadata.name);

            mainNetworkDescription = std::move(networkDescription);
            break;
        }
    } break;
    default:
        OPENVINO_THROW("Invalid \"SEPARATE_WEIGHTS_VERSION\" value found within the \"compileWS\" call:",
                       config.get<SEPARATE_WEIGHTS_VERSION>());
        break;
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        auto compile_model_mem_end = get_peak_memory_usage();
        _logger.debug("Start of compilation memory usage: Peak %lld KB", compile_model_mem_start);
        _logger.debug("End of compilation memory usage: Peak %lld KB", compile_model_mem_end);
        // Note: Following log is parsed by CI. Take care when modifying it.
        _logger.info("Compilation memory usage: Peak %lld KB", compile_model_mem_end - compile_model_mem_start);
    }

    _logger.debug("compile end");

    ov::Tensor tensorMain = make_tensor_from_vector(mainNetworkDescription->compiledNetwork);
    ze_graph_handle_t mainGraphHandle = nullptr;
    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to
        // get the graph handle from the compiled network
        try {
            mainGraphHandle = _zeGraphExt->getGraphHandle(*reinterpret_cast<const uint8_t*>(tensorMain.data()),
                                                          tensorMain.get_byte_size());
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }

    std::vector<ze_graph_handle_t> initGraphHandles;
    std::vector<ov::Tensor> tensorsInits;
    std::vector<NetworkMetadata> initNetworkMetadata;
    initGraphHandles.reserve(initNetworkDescriptions.size());
    tensorsInits.reserve(initNetworkDescriptions.size());
    initNetworkMetadata.reserve(initNetworkDescriptions.size());
    for (auto& networkDesc : initNetworkDescriptions) {
        ov::Tensor tensor = make_tensor_from_vector(networkDesc->compiledNetwork);
        ze_graph_handle_t graphHandle = nullptr;
        if (_zeGraphExt) {
            try {
                graphHandle = _zeGraphExt->getGraphHandle(*reinterpret_cast<const uint8_t*>(tensor.data()),
                                                          tensor.get_byte_size());
            } catch (...) {
            }
        }

        initGraphHandles.push_back(graphHandle);
        tensorsInits.push_back(std::move(tensor));
        initNetworkMetadata.push_back(std::move(networkDesc->metadata));
    }

    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             /* persistentBlob = */ true,
                                             mainGraphHandle,
                                             std::move(mainNetworkDescription->metadata),
                                             std::move(tensorMain),
                                             initGraphHandles,
                                             std::move(initNetworkMetadata),
                                             tensorsInits,
                                             model,
                                             config,
                                             _compiler);
}

std::shared_ptr<IGraph> PluginCompilerAdapter::parse(
    ov::Tensor mainBlob,
    const Config& config,
    std::optional<std::vector<ov::Tensor>> initBlobs,
    const std::optional<std::shared_ptr<const ov::Model>>& model) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "parse");

    _logger.debug("parse start");
    std::vector<uint8_t> network(mainBlob.get_byte_size());
    network.assign(reinterpret_cast<const uint8_t*>(mainBlob.data()),
                   reinterpret_cast<const uint8_t*>(mainBlob.data()) + mainBlob.get_byte_size());
    auto networkMeta = _compiler->parse(network, config);
    network.clear();
    network.shrink_to_fit();

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        graphHandle =
            _zeGraphExt->getGraphHandle(*reinterpret_cast<const uint8_t*>(mainBlob.data()), mainBlob.get_byte_size());
    }

    _logger.debug("main schedule parse end");

    const bool persistentBlob = config.get<LOADED_FROM_CACHE>();

    if (!initBlobs.has_value()) {
        return std::make_shared<Graph>(_zeGraphExt,
                                       _zeroInitStruct,
                                       graphHandle,
                                       std::move(networkMeta),
                                       std::move(mainBlob),
                                       persistentBlob,
                                       config,
                                       _compiler);
    }

    // The presence of init schedules means weights separation has been enabled at compilation time. Use a specific
    // "Graph" object as wrapper over all L0 handles.
    std::vector<ze_graph_handle_t> initGraphHandles;
    std::vector<NetworkMetadata> initMetadata;
    for (const auto& initBlob : initBlobs.value()) {
        network.reserve(initBlob.get_byte_size());
        network.assign(reinterpret_cast<const uint8_t*>(initBlob.data()),
                       reinterpret_cast<const uint8_t*>(initBlob.data()) + initBlob.get_byte_size());
        initMetadata.push_back(_compiler->parse(network, config));
        network.clear();
        network.shrink_to_fit();

        if (_zeGraphExt) {
            initGraphHandles.push_back(_zeGraphExt->getGraphHandle(*reinterpret_cast<const uint8_t*>(initBlob.data()),
                                                                   initBlob.get_byte_size()));
        }
    }

    _logger.debug("init schedules parse end");
    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             persistentBlob,
                                             graphHandle,
                                             std::move(networkMeta),
                                             std::move(mainBlob),
                                             initGraphHandles,
                                             std::move(initMetadata),
                                             std::move(initBlobs),
                                             model.value(),
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
