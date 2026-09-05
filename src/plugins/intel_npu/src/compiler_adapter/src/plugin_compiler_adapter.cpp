// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <memory>
#include <string>

#include "compiler_impl.hpp"
#include "dynamic_graph.hpp"
#include "graph.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "mem_usage.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "weightless_graph.hpp"
#include "weightless_utils.hpp"

namespace intel_npu {

namespace {
constexpr OptionSupportCache::CacheKey pluginOptionSupportKey =
    static_cast<OptionSupportCache::CacheKey>(ov::intel_npu::CompilerType::PLUGIN);

/// Loads the compiler-in-plugin, translating any failure into the aborting message callers expect.
ov::SoPtr<IVCLCompiler> loadVCLCompiler(const std::optional<IDevice::DeviceProperties>& deviceProperties) {
    Logger logger("PluginCompilerAdapter", Logger::global().level());
    logger.info("Loading PLUGIN compiler");
    try {
        return makeVCLCompiler(ov::util::path_to_string(ov::util::get_ov_lib_path()), deviceProperties);
    } catch (const std::exception& vclException) {
        OPENVINO_THROW("VCL compiler loading failed, aborting. Error: ", vclException.what());
    }
}
}  // namespace

PluginCompilerAdapter::PluginCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                                             const std::shared_ptr<OptionSupportCache>& optionSupportCache,
                                             const std::optional<IDevice::DeviceProperties>& deviceProperties)
    : PluginCompilerAdapter(loadVCLCompiler(deviceProperties), zeroInitStruct, optionSupportCache) {}

PluginCompilerAdapter::PluginCompilerAdapter(ov::SoPtr<IVCLCompiler> compiler,
                                             const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                                             const std::shared_ptr<OptionSupportCache>& optionSupportCache)
    : _zeroInitStruct(zeroInitStruct),
      _optionSupportCache(optionSupportCache),
      _compiler(std::move(compiler)),
      _logger("PluginCompilerAdapter", Logger::global().level()) {
    _logger.info("initialize PluginCompilerAdapter start");

    OPENVINO_ASSERT(_compiler != nullptr, "PluginCompilerAdapter requires a non-null compiler");

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
    auto [tensor, compatibilityDescriptor] = _compiler->compile(model, config);
    _logger.debug("compile end");

    const auto& compilationMode = config.get<COMPILATION_MODE>();
    const bool isHostCompile = compilationMode.find("HostCompile") != std::string::npos;
    const BlobType blobType =
        isHostCompile ? (compilationMode.find("HostCompile_Interpreter") != std::string::npos ? BlobType::BYTECODE
                                                                                              : BlobType::LLVM)
                      : BlobType::ELF;
    if (blobType != BlobType::ELF) {
        _logger.debug("HostCompile mode is detected from NPU_COMPILATION_MODE, use internal function to get metadata!");
        NPUVMRuntimeApi::initializeFromBlob(tensor.data(), tensor.get_byte_size());

        // metadata will be obtained in initialze() of DynamicGraph
        _logger.debug("Use dynamicGraph to hold blob for HostCompile mode!");
        return std::make_shared<DynamicGraph>(_zeroInitStruct, std::move(tensor), config, blobType);
    }

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
        compatibilityDescriptor,
        /* persistentBlob = */ true);  // exporting the blob shall be available in such a scenario
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compileWS(std::shared_ptr<ov::Model>&& model,
                                                         const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compileWS");
    _logger.debug("compile start");

    FilteredConfig localConfig = config;
    if (!localConfig.has<SEPARATE_WEIGHTS_VERSION>()) {
        localConfig.update({{ov::intel_npu::separate_weights_version.name(), "ONE_SHOT"}});
    }

    _logger.info("SEPARATE_WEIGHTS_VERSION: %s",
                 SEPARATE_WEIGHTS_VERSION::toString(localConfig.get<SEPARATE_WEIGHTS_VERSION>()).c_str());

    int64_t compileModelMemStart = 0;
    if (_logger.level() >= ov::log::Level::INFO) {
        compileModelMemStart = get_peak_memory_usage();
    }

    std::vector<ov::Tensor> tensorsInits;
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initNetworkMetadata;

    ov::Tensor tensorMain;
    GraphDescriptor mainGraphDesc;
    NetworkMetadata mainNetworkMetadata;
    std::optional<std::string> compatibilityDescriptor;

    switch (localConfig.get<SEPARATE_WEIGHTS_VERSION>()) {
    case ov::intel_npu::WSVersion::ONE_SHOT: {
        auto oneShotResult = _compiler->compileWsOneShot(model, localConfig);
        auto initMainTensors = std::move(oneShotResult.first);
        compatibilityDescriptor = std::move(oneShotResult.second);

        tensorMain = initMainTensors.back();
        initMainTensors.pop_back();
        if (initMainTensors.empty()) {
            _logger.warning("NPU compiler did not produce any init schedules. "
                            "This likely means that the compiled model blob has weights inside even "
                            "though weightless compilation was requested.");
        }

        tensorsInits = std::move(initMainTensors);

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

        initGraphDescriptors.reserve(tensorsInits.size());
        initNetworkMetadata.reserve(tensorsInits.size());
        for (const auto& tensor : tensorsInits) {
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

        while (true) {
            auto iterativeResult = _compiler->compileWsIterative(targetModel, localConfig, i++);
            auto tensor = std::move(iterativeResult.first);
            if (iterativeResult.second.has_value()) {
                compatibilityDescriptor = std::move(iterativeResult.second);
            }
            if (!tensor) {
                break;
            }
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
        auto compileModelMemEnd = get_peak_memory_usage();
        _logger.debug("Start of compilation memory usage: Peak %lld KB", compileModelMemStart);
        _logger.debug("End of compilation memory usage: Peak %lld KB", compileModelMemEnd);
        // Note: Following log is parsed by CI. Take care when modifying it.
        _logger.info("Compilation memory usage: Peak %lld KB", compileModelMemEnd - compileModelMemStart);
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
        std::move(model),
        localConfig,
        /* persistentBlob = */ true,
        compatibilityDescriptor);  // exporting the blob shall be available in such a scenario
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
    // Trimming and tokenisation happen in the compiler; the adapter only owns the cache write-through.
    const std::vector<std::string> compilerOpts = _compiler->get_supported_options();

    if (_optionSupportCache) {
        _optionSupportCache->setSupportedOptions(pluginOptionSupportKey, compilerOpts);
    }
    return compilerOpts;
}

bool PluginCompilerAdapter::is_option_supported(const std::string& optname,
                                                const std::optional<std::string>& optValue) const {
    bool optionSupportCache = _optionSupportCache && !optValue.has_value();
    if (optionSupportCache) {
        const auto cachedSupport = _optionSupportCache->isOptionSupported(pluginOptionSupportKey, optname);
        if (cachedSupport.has_value()) {
            return cachedSupport.value();
        }
    }

    const bool supported = _compiler->is_option_supported(optname, optValue);
    if (optionSupportCache) {
        _optionSupportCache->addSupportedOption(pluginOptionSupportKey, optname, supported);
    }

    const char* valueForLog = optValue.has_value() ? optValue->c_str() : "null";
    _logger.debug("Option %s %s `%s` by VCLCompilerImpl",
                  optname.c_str(),
                  supported ? "is supported" : "is not supported",
                  valueForLog);

    return supported;
}

}  // namespace intel_npu
