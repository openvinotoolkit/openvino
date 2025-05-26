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
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "mem_usage.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "weightless_graph.hpp"


namespace {
#ifndef VCL_FOR_COMPILER
std::shared_ptr<void> load_library(const std::string& libpath) {
#    if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    return ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#    else
    return ov::util::load_shared_object(libpath.c_str());
#    endif
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
#endif
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

#ifdef VCL_FOR_COMPILER
    _logger.info("VCL driver compiler will be used.");
    _compiler = ov::SoPtr<intel_npu::ICompiler>(VCLCompilerImpl::getInstance(), VCLApi::getInstance()->getLibrary());
#else
    _logger.info("MLIR compiler will be used.");
    std::string baseName = "npu_mlir_compiler";
    auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
    _compiler = load_compiler(libPath);
#endif
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
    GraphDescriptor graphDesc;

    NetworkMetadata networkMeta;
    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to get the graph handle from the compiled
        // network
        try {
            graphDesc =
                _zeGraphExt->getGraphDescriptor(tensor.data(), tensor.get_byte_size());
#ifdef VCL_FOR_COMPILER
            // For VCL, we need to get metadata from driver parser
            networkMeta = _zeGraphExt->getNetworkMeta(graphDesc);
            networkMeta.name = model->get_friendly_name();
>>>>>>> e20458aedf (Add VCLApi and VCLCompilerImpl)
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }

    return std::make_shared<Graph>(_zeGraphExt,
                                   _zeroInitStruct,
                                   graphDesc,
#ifdef VCL_FOR_COMPILER
                                   std::move(networkMeta),
#else
                                   std::move(networkDesc.metadata),
#endif
                                   std::move(tensor),
                                   /* blobAllocatedByPlugin = */ false,
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

    Config localConfig = config;
    if (!localConfig.has<SEPARATE_WEIGHTS_VERSION>()) {
        localConfig.update({{ov::intel_npu::separate_weights_version.name(), "ONE_SHOT"}});
    }

    _logger.info("SEPARATE_WEIGHTS_VERSION: %s",
                 SEPARATE_WEIGHTS_VERSION::toString(localConfig.get<SEPARATE_WEIGHTS_VERSION>()).c_str());

    int64_t compile_model_mem_start = 0;
    if (_logger.level() >= ov::log::Level::INFO) {
        compile_model_mem_start = get_peak_memory_usage();
    }
    switch (localConfig.get<SEPARATE_WEIGHTS_VERSION>()) {
    case ov::intel_npu::WSVersion::ONE_SHOT: {
        std::vector<std::shared_ptr<NetworkDescription>> initMainNetworkDescriptions =
            _compiler->compileWsOneShot(model, localConfig);

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
                   std::make_shared<NetworkDescription>(_compiler->compileWsIterative(targetModel, localConfig, i++))) {
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
        OPENVINO_THROW("Invalid \"SEPARATE_WEIGHTS_VERSION\" value found within the \"compileWS\" call: ",
                       localConfig.get<SEPARATE_WEIGHTS_VERSION>());
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
    GraphDescriptor mainGraphDesc;
    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to
        // get the graph handle from the compiled network
        try {
            mainGraphDesc = _zeGraphExt->getGraphDescriptor(tensorMain.data(), tensorMain.get_byte_size());
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }

    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<ov::Tensor> tensorsInits;
    std::vector<NetworkMetadata> initNetworkMetadata;
    initGraphDescriptors.reserve(initNetworkDescriptions.size());
    tensorsInits.reserve(initNetworkDescriptions.size());
    initNetworkMetadata.reserve(initNetworkDescriptions.size());
    for (auto& networkDesc : initNetworkDescriptions) {
        ov::Tensor tensor = make_tensor_from_vector(networkDesc->compiledNetwork);
        GraphDescriptor initGraphDesc;
        if (_zeGraphExt) {
            try {
                initGraphDesc = _zeGraphExt->getGraphDescriptor(tensor.data(), tensor.get_byte_size());
            } catch (...) {
            }
        }

        initGraphDescriptors.push_back(initGraphDesc);
        tensorsInits.push_back(std::move(tensor));
        initNetworkMetadata.push_back(std::move(networkDesc->metadata));
    }

    return std::make_shared<WeightlessGraph>(
        _zeGraphExt,
        _zeroInitStruct,
        mainGraphDesc,
        std::move(mainNetworkDescription->metadata),
        std::move(tensorMain),
        initGraphDescriptors,
        std::move(initNetworkMetadata),
        tensorsInits,
        model,
        localConfig,
        /* persistentBlob = */ true,  // exporting the blob shall be available in such a scenario
        _compiler);
}

std::shared_ptr<IGraph> PluginCompilerAdapter::parse(
    ov::Tensor mainBlob,
    const Config& config,
    std::optional<std::vector<ov::Tensor>> initBlobs,
    const std::optional<std::shared_ptr<const ov::Model>>& model) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "parse");

    ze_graph_handle_t graphHandle = nullptr;
    NetworkMetadata networkMeta;
    std::vector<uint8_t> network(mainBlob.get_byte_size());

#ifdef VCL_FOR_COMPILER
    _logger.debug("parse metadata from driver for vcl compiler");
    if (_zeGraphExt) {
        _logger.debug("parse start for vcl compiler");
        graphHandle = _zeGraphExt->getGraphHandle(*reinterpret_cast<const uint8_t*>(mainBlob.data()), mainBlob.get_byte_size());
        networkMeta = _zeGraphExt->getNetworkMeta(graphHandle);
    }
    _logger.debug("parse end for vcl compiler");
#else
    _logger.debug("parse start");
    network.assign(reinterpret_cast<const uint8_t*>(mainBlob.data()),
                   reinterpret_cast<const uint8_t*>(mainBlob.data()) + mainBlob.get_byte_size());
    auto networkMeta = _compiler->parse(network, config);
    network.clear();
    network.shrink_to_fit();

<<<<<<< HEAD
    GraphDescriptor mainGraphDesc;

=======
>>>>>>> e20458aedf (Add VCLApi and VCLCompilerImpl)
    if (_zeGraphExt) {
        mainGraphDesc = _zeGraphExt->getGraphDescriptor(mainBlob.data(), mainBlob.get_byte_size());
    }

    _logger.debug("main schedule parse end");
#endif

    // exporting the blob when we get it from cache or ov::hint::compiled_blob property
    // shall be available
    const bool blobIsPersistent = config.has<COMPILED_BLOB>()       ? true
                                  : config.has<LOADED_FROM_CACHE>() ? config.get<LOADED_FROM_CACHE>()
                                                                    : false;

    if (!initBlobs.has_value()) {
        return std::make_shared<Graph>(_zeGraphExt,
                                       _zeroInitStruct,
                                       mainGraphDesc,
                                       std::move(networkMeta),
                                       std::move(mainBlob),
                                       config,
                                       blobIsPersistent,
                                       _compiler);
    }

    // The presence of init schedules means weights separation has been enabled at compilation time. Use a specific
    // "Graph" object as wrapper over all L0 handles.
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initMetadata;

    for (const auto& initBlob : initBlobs.value()) {
        network.reserve(initBlob.get_byte_size());
        network.assign(reinterpret_cast<const uint8_t*>(initBlob.data()),
                       reinterpret_cast<const uint8_t*>(initBlob.data()) + initBlob.get_byte_size());
        initMetadata.push_back(_compiler->parse(network, config));
        network.clear();
        network.shrink_to_fit();

        if (_zeGraphExt) {
            auto initGraphDesc = _zeGraphExt->getGraphDescriptor(initBlob.data(), initBlob.get_byte_size());

            initGraphDescriptors.push_back(initGraphDesc);
        }
    }

    _logger.debug("init schedules parse end");
    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             mainGraphDesc,
                                             std::move(networkMeta),
                                             std::move(mainBlob),
                                             initGraphDescriptors,
                                             std::move(initMetadata),
                                             std::move(initBlobs),
                                             model.value(),
                                             config,
                                             blobIsPersistent,
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
#ifdef VCL_FOR_COMPILER
    // For VCL, we can return the supported options from compiler
    VCLCompilerImpl* vclCompiler = dynamic_cast<VCLCompilerImpl*>(_compiler.operator->());
    if (vclCompiler == nullptr) {
        _logger.warning("Failed to cast compiler to VCLCompilerImpl. Returning empty supported options.");
        return {};
    }
    std::vector<char> options;
    if (!vclCompiler->get_supported_options(options)) {
        _logger.warning("VCLCompilerImpl get_supported_options failed. Returning empty supported options.");
        return {};
    }

    if (options.empty()) {
        _logger.warning("get_supported_options returned empty options.");
        return {};
    }

    std::string compilerOptionsStr(options.data(), options.size());
    _logger.debug("VCLCompilerImpl return supported_options: %s", compilerOptionsStr.c_str());
    // vectorize string
    std::istringstream suppstream(compilerOptionsStr);
    std::vector<std::string> compilerOpts = {};
    std::string option;
    while (suppstream >> option) {
        compilerOpts.push_back(option);
    }
    return compilerOpts;
#else
    // PluginCompiler has all the same options as plugin
    // Returing empty string to let the plugin fallback to legacy registration
    return {};
#endif
}

bool PluginCompilerAdapter::is_option_supported(std::string optname) const {
#ifdef VCL_FOR_COMPILER
    VCLCompilerImpl* vclCompiler = dynamic_cast<VCLCompilerImpl*>(_compiler.operator->());
    if (vclCompiler == nullptr) {
        _logger.warning("Failed to cast compiler to VCLCompilerImpl. Returning false for check.");
        return false;
    }
    if (vclCompiler->is_option_supported(optname)) {
        _logger.debug("Option %s is supported by VCLCompilerImpl", optname.c_str());
        return true;
    } else {
        _logger.debug("Option %s is not supported by VCLCompilerImpl", optname.c_str());
        return false;
    }
#else
    // This functions has no utility in PluginCompiler
    // returning false for any request to avoid the option of spaming the plugin
    return false;
#endif
}

}  // namespace intel_npu
