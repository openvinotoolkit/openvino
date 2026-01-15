// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <memory>
#include <string>

#include "compiler_impl.hpp"
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
#include "weightless_utils.hpp"

namespace {

std::shared_ptr<intel_npu::ICompiler> get_compiler(std::shared_ptr<void> so) {
    static constexpr auto CreateFuncName = "CreateNPUCompiler";
    auto symbol = ov::util::get_symbol(so, CreateFuncName);

    using CreateFuncT = void (*)(std::shared_ptr<intel_npu::ICompiler>&);
    const auto createFunc = reinterpret_cast<CreateFuncT>(symbol);

    std::shared_ptr<intel_npu::ICompiler> compilerPtr;
    createFunc(compilerPtr);
    return compilerPtr;
}

ov::SoPtr<intel_npu::ICompiler> load_compiler(const std::filesystem::path& libpath) {
    auto compilerSO = ov::util::load_shared_object(libpath);
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

    _logger.info("Loading PLUGIN compiler");
    try {
        auto vclCompilerPtr = VCLCompilerImpl::getInstance();
        OPENVINO_ASSERT(vclCompilerPtr != nullptr, "VCL compiler is nullptr");
        auto vclLib = vclCompilerPtr->getLinkedLibrary();
        _logger.info("PLUGIN VCL compiler is loading");
        OPENVINO_ASSERT(vclLib != nullptr, "VCL library is nullptr");
        _compiler = ov::SoPtr<intel_npu::ICompiler>(vclCompilerPtr, vclLib);
    } catch (const std::exception& vcl_exception) {
        _logger.info("VCL compiler load failed: %s. Trying to load MLIR compiler...", vcl_exception.what());
        std::string baseName = "npu_mlir_compiler";
        auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
        try {
            _compiler = load_compiler(libPath);
            if (!_compiler) {
                throw std::runtime_error("MLIR compiler load returned nullptr");
            } else {
                _logger.info("MLIR compiler loaded successfully. PLUGIN compiler will be used.");
            }
        } catch (const std::exception& mlir_exception) {
            _logger.info("MLIR compiler load failed: %s", mlir_exception.what());
            throw std::runtime_error("Both VCL and MLIR compiler load failed, aborting.");
        }
    }

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
                                                       const FilteredConfig& config) const {
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
            graphDesc = _zeGraphExt->getGraphDescriptor(tensor.data(), tensor.get_byte_size());
            networkMeta = _zeGraphExt->getNetworkMeta(graphDesc);
            networkMeta.name = model->get_friendly_name();
        } catch (const std::exception& ex) {
            _logger.info("Failed to use the level zero graph handle: %s. Inference requests for this model are not "
                         "allowed. Only exports are available",
                         ex.what());
        }
    } else {
        _logger.warning("No driver is found, zeGraphExt is nullptr, so metadata is empty. Only exports are available");
    }

    return std::make_shared<Graph>(
        _zeGraphExt,
        _zeroInitStruct,
        graphDesc,
        std::move(networkMeta),
        std::move(tensor),
        config,
        /* persistentBlob = */ true,  // exporting the blob shall be available in such a scenario
        _compiler);
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compileWS(const std::shared_ptr<ov::Model>& model,
                                                         const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compileWS");
    _logger.debug("compile start");

    FilteredConfig localConfig = config;
    if (!localConfig.has<SEPARATE_WEIGHTS_VERSION>()) {
        localConfig.update({{ov::intel_npu::separate_weights_version.name(), "ONE_SHOT"}});
    }

    _logger.info("SEPARATE_WEIGHTS_VERSION: %s",
                 SEPARATE_WEIGHTS_VERSION::toString(localConfig.get<SEPARATE_WEIGHTS_VERSION>()).c_str());

    int64_t compile_model_mem_start = 0;
    if (_logger.level() >= ov::log::Level::INFO) {
        compile_model_mem_start = get_peak_memory_usage();
    }

    std::vector<ov::Tensor> tensorsInits;
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initNetworkMetadata;

    ov::Tensor tensorMain;
    GraphDescriptor mainGraphDesc;
    NetworkMetadata mainNetworkMetadata;

    switch (localConfig.get<SEPARATE_WEIGHTS_VERSION>()) {
    case ov::intel_npu::WSVersion::ONE_SHOT: {
        std::vector<std::shared_ptr<NetworkDescription>> initMainNetworkDescriptions =
            _compiler->compileWsOneShot(model, localConfig);

        std::shared_ptr<NetworkDescription> mainNetworkDescription = initMainNetworkDescriptions.back();
        initMainNetworkDescriptions.pop_back();
        OPENVINO_ASSERT(initMainNetworkDescriptions.size() > 0, "No init schedules have been returned by the compiler");
        std::vector<std::shared_ptr<NetworkDescription>> initNetworkDescriptions =
            std::move(initMainNetworkDescriptions);

        tensorMain = make_tensor_from_vector(mainNetworkDescription->compiledNetwork);
        if (_zeGraphExt) {
            // Depending on the config, we may get an error when trying to
            // get the graph handle from the compiled network
            try {
                mainGraphDesc = _zeGraphExt->getGraphDescriptor(tensorMain.data(), tensorMain.get_byte_size());
                mainNetworkMetadata = _zeGraphExt->getNetworkMeta(mainGraphDesc);
            } catch (const std::exception& ex) {
                _logger.info("Failed to use the level zero graph handle: %s. Inference requests for this model are not "
                             "allowed. Only exports are available",
                             ex.what());
            }
        } else {
            _logger.warning(
                "No driver is found, zeGraphExt is nullptr, so metadata is empty. Only exports are available");
        }

        initGraphDescriptors.reserve(initNetworkDescriptions.size());
        tensorsInits.reserve(initNetworkDescriptions.size());
        initNetworkMetadata.reserve(initNetworkDescriptions.size());
        for (auto& networkDesc : initNetworkDescriptions) {
            ov::Tensor tensor = make_tensor_from_vector(networkDesc->compiledNetwork);
            GraphDescriptor initGraphDesc;
            NetworkMetadata initNetworkMeta;
            if (_zeGraphExt) {
                try {
                    initGraphDesc = _zeGraphExt->getGraphDescriptor(tensor.data(), tensor.get_byte_size());
                    initNetworkMeta = _zeGraphExt->getNetworkMeta(initGraphDesc);
                } catch (const std::exception& ex) {
                    _logger.info(
                        "Failed to use the level zero graph handle: %s. Inference requests for this model are not "
                        "allowed. Only exports are available",
                        ex.what());
                }
            } else {
                _logger.warning(
                    "No driver is found, zeGraphExt is nullptr, so metadata is empty. Only exports are available");
            }

            initGraphDescriptors.push_back(initGraphDesc);
            tensorsInits.push_back(std::move(tensor));
            initNetworkMetadata.push_back(std::move(initNetworkMeta));
        }
    } break;
    case ov::intel_npu::WSVersion::ITERATIVE: {
        OPENVINO_ASSERT(_zeGraphExt,
                        "The \"iterative\" implementation of the weights separation feature requires a Level Zero "
                        "graph handle to compile a model.");

        // The state of the model needs to be reset every iteration
        const std::shared_ptr<ov::Model> originalModel = model->clone();
        std::shared_ptr<ov::Model> targetModel = model;
        size_t i = 0;

        while (auto networkDescription =
                   std::make_shared<NetworkDescription>(_compiler->compileWsIterative(targetModel, localConfig, i++))) {
            ov::Tensor tensor = make_tensor_from_vector(networkDescription->compiledNetwork);
            GraphDescriptor graphDesc = _zeGraphExt->getGraphDescriptor(tensor.data(), tensor.get_byte_size());
            NetworkMetadata networkMetadata = _zeGraphExt->getNetworkMeta(graphDesc);

            if (isInitMetadata(networkMetadata)) {
                networkMetadata.name = model->get_friendly_name() + "_init";
                targetModel = originalModel->clone();
                initGraphDescriptors.push_back(graphDesc);
                tensorsInits.push_back(std::move(tensor));
                initNetworkMetadata.push_back(std::move(networkMetadata));
                continue;
            }

            networkMetadata.name = model->get_friendly_name() + "_main";
            tensorMain = std::move(tensor);
            mainGraphDesc = graphDesc;
            mainNetworkMetadata = std::move(networkMetadata);
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

    return std::make_shared<WeightlessGraph>(
        _zeGraphExt,
        _zeroInitStruct,
        mainGraphDesc,
        std::move(mainNetworkMetadata),
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
    const ov::Tensor& mainBlob,
    const FilteredConfig& config,
    const std::optional<std::vector<ov::Tensor>>& initBlobs,
    const std::optional<std::shared_ptr<const ov::Model>>& model) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "parse");

    GraphDescriptor mainGraphDesc;
    NetworkMetadata mainNetworkMetadata;

    if (_zeGraphExt) {
        _logger.debug("parse start");
        mainGraphDesc = _zeGraphExt->getGraphDescriptor(mainBlob.data(), mainBlob.get_byte_size());
        mainNetworkMetadata = _zeGraphExt->getNetworkMeta(mainGraphDesc);
        _logger.debug("main schedule parse end");
        if (model) {
            mainNetworkMetadata.name = model.value()->get_friendly_name();
        } else {
            _logger.info("networkMeta name is empty in parse!");
        }
    } else {
        _logger.warning("no zeGraphExt, metadata is empty from vcl compiler.");
    }

    // exporting the blob when we get it from cache or ov::hint::compiled_blob property
    // shall be available
    const bool blobIsPersistent = config.has<COMPILED_BLOB>()       ? true
                                  : config.has<LOADED_FROM_CACHE>() ? config.get<LOADED_FROM_CACHE>()
                                                                    : false;

    if (!initBlobs.has_value()) {
        return std::make_shared<Graph>(_zeGraphExt,
                                       _zeroInitStruct,
                                       mainGraphDesc,
                                       std::move(mainNetworkMetadata),
                                       mainBlob,
                                       config,
                                       blobIsPersistent,
                                       _compiler);
    }

    // The presence of init schedules means weights separation has been enabled at compilation time. Use a specific
    // "Graph" object as wrapper over all L0 handles.
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initNetworkMetadata;

    for (const auto& initBlob : initBlobs.value()) {
        if (_zeGraphExt) {
            auto initGraphDesc = _zeGraphExt->getGraphDescriptor(initBlob.data(), initBlob.get_byte_size());
            auto initNetworkMeta = _zeGraphExt->getNetworkMeta(initGraphDesc);

            initGraphDescriptors.push_back(initGraphDesc);
            initNetworkMetadata.push_back(std::move(initNetworkMeta));
        }
    }

    _logger.debug("init schedules parse end");
    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             mainGraphDesc,
                                             std::move(mainNetworkMetadata),
                                             mainBlob,
                                             initGraphDescriptors,
                                             std::move(initNetworkMetadata),
                                             initBlobs,
                                             model.value(),
                                             config,
                                             blobIsPersistent,
                                             _compiler);
}

ov::SupportedOpsMap PluginCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "query");

    return _compiler->query(model, config);
}

uint32_t PluginCompilerAdapter::get_version() const {
    // returning max val as PluginCompiler supports all features and options the plugin is aware of
    return _compiler->get_version();
}

std::vector<std::string> PluginCompilerAdapter::get_supported_options() const {
    // For VCL, we can return the supported options from compiler
    VCLCompilerImpl* vclCompiler = dynamic_cast<VCLCompilerImpl*>(_compiler.operator->());
    if (vclCompiler == nullptr) {
        // If _compiler  cannot be cast to VCLCompilerImpl, it should use the mlir library.
        // PluginCompiler has all the same options as plugin
        // Returing empty string to let the plugin fallback to legacy registration
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
}

bool PluginCompilerAdapter::is_option_supported(std::string optname, std::optional<std::string> optValue) const {
    VCLCompilerImpl* vclCompiler = dynamic_cast<VCLCompilerImpl*>(_compiler.operator->());
    if (vclCompiler == nullptr) {
        // If _compiler  cannot be cast to VCLCompilerImpl, it should use the mlir library.
        // This functions has no utility in PluginCompiler
        // returning false for any request to avoid the option of spamming the plugin
        _logger.warning("Failed to cast compiler to VCLCompilerImpl. Returning false for check.");
        return false;
    }

    const char* optvalue_ch = optValue.has_value() ? optValue.value().c_str() : nullptr;
    if (vclCompiler->is_option_supported(optname, optValue)) {
        _logger.debug("Option %s is supported `%s` by VCLCompilerImpl",
                      optname.c_str(),
                      optvalue_ch ? optvalue_ch : "null");
        return true;
    } else {
        _logger.debug("Option %s is not supported `%s` by VCLCompilerImpl",
                      optname.c_str(),
                      optvalue_ch ? optvalue_ch : "null");
        return false;
    }
}

}  // namespace intel_npu
