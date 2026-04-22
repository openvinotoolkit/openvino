// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include <functional>
#include <string_view>

#include "graph.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "mem_usage.hpp"
#include "model_serializer.hpp"
#include "openvino/core/model.hpp"
#include "weightless_graph.hpp"
#include "weightless_utils.hpp"

namespace intel_npu {

DriverCompilerAdapter::DriverCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("DriverCompilerAdapter", Logger::global().level()) {
    _logger.info("initialize DriverCompilerAdapter start");

    uint32_t graphExtVersion = _zeroInitStruct->getGraphDdiTable().version();

    _compilerProperties = _zeroInitStruct->getCompilerProperties();

    _logger.debug("DriverCompilerAdapter creating adapter using graphExtVersion");

    _zeGraphExt = std::make_shared<ZeGraphExtWrappers>(_zeroInitStruct);

    _logger.info("initialize DriverCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compile");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");

    const auto isOptionValueSupportedByCompiler = [this](const std::string& optionName,
                                                         const std::optional<std::string>& optionValue) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionValueSupportedByCompiler,
                                                    _zeGraphExt->isPluginModelHashSupported());
    FilteredConfig updatedConfig = config;
    if (config.isAvailable(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update({{ov::intel_npu::model_serializer_version.name(),
                               MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion)}});
    }

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));
    const auto isOptionSupportedByCompiler = std::bind(&DriverCompilerAdapter::isCompilerOptionSupported,
                                                       this,
                                                       std::cref(updatedConfig),
                                                       std::cref(compilerVersion),
                                                       std::placeholders::_1);

    _logger.debug("build flags");
    buildFlags += compiler_utils::serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += compiler_utils::serializeConfig(updatedConfig, compilerVersion, isOptionSupportedByCompiler);

    _logger.debug("compileIR Build flags : %s", buildFlags.c_str());

    _logger.debug("compile start");
    // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
    const bool bypassCache = !updatedConfig.get<CACHE_DIR>().empty() || updatedConfig.get<BYPASS_UMD_CACHING>();
    auto graphDesc = _zeGraphExt->getGraphDescriptor(std::move(serializedIR), buildFlags, bypassCache);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(graphDesc);
    networkMeta.name = model->get_friendly_name();

    std::optional<std::string> compatibilityDescriptor;
    if (_zeGraphExt->isCompatibilityDescriptorSupported()) {
        compatibilityDescriptor = _zeGraphExt->getCompatibilityDescriptor(graphDesc._handle);
    }

    return std::make_shared<Graph>(_zeGraphExt,
                                   _zeroInitStruct,
                                   graphDesc,
                                   std::move(networkMeta),
                                   /* blob = */ std::nullopt,
                                   updatedConfig,
                                   compatibilityDescriptor);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compileWS(std::shared_ptr<ov::Model>&& model,
                                                         const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compileWS");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    if ((compilerVersion.major < 6) || (compilerVersion.major == 6 && compilerVersion.minor < 3)) {
        OPENVINO_THROW("Minimum compiler version required for weights separation: 6.3. Found: ",
                       compilerVersion.major,
                       ".",
                       compilerVersion.minor);
    }

    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    if (config.get<SEPARATE_WEIGHTS_VERSION>() != ov::intel_npu::WSVersion::ITERATIVE) {
        OPENVINO_THROW("Invalid \"SEPARATE_WEIGHTS_VERSION\" value found within the \"compileWS\" call:",
                       config.get<SEPARATE_WEIGHTS_VERSION>(),
                       ". \"WSVersion::ITERATIVE\" is the only supported value for the compiler-in-driver path.");
    }

    _logger.debug("serialize IR");
    const auto isOptionValueSupportedByCompiler = [this](const std::string& optionName,
                                                         const std::optional<std::string>& optionValue) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionValueSupportedByCompiler,
                                                    _zeGraphExt->isPluginModelHashSupported(),
                                                    true);
    FilteredConfig updatedConfig = config;
    if (config.isAvailable(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update({{ov::intel_npu::model_serializer_version.name(),
                               MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion)}});
    }

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    const std::string serializedIOInfo = compiler_utils::serializeIOInfo(model, useIndices);

    // WS v3 is based on a stateless compiler. We'll use a separate config entry for informing the compiler the index of
    // the current call iteration.
    std::vector<NetworkMetadata> initNetworkMetadata;
    NetworkMetadata mainNetworkMetadata;
    std::vector<GraphDescriptor> initGraphDescriptors;
    GraphDescriptor mainGraphHandle;
    size_t callNumber = 0;

    // Convention: run until the main schedule has been returned.
    int64_t compile_model_mem_start = 0;
    if (_logger.level() >= ov::log::Level::INFO) {
        compile_model_mem_start = get_peak_memory_usage();
    }

    const auto isOptionSupportedByCompiler = std::bind(&DriverCompilerAdapter::isCompilerOptionSupported,
                                                       this,
                                                       std::cref(updatedConfig),
                                                       std::cref(compilerVersion),
                                                       std::placeholders::_1);

    while (true) {
        _logger.debug("compileWS iteration %d", callNumber);
        updatedConfig.update({{ov::intel_npu::ws_compile_call_number.name(), std::to_string(callNumber++)}});

        _logger.debug("build flags");
        buildFlags = serializedIOInfo;
        buildFlags += " ";
        buildFlags += compiler_utils::serializeConfig(updatedConfig, compilerVersion, isOptionSupportedByCompiler);

        _logger.debug("compile start");
        // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
        const bool bypassCache = !updatedConfig.get<CACHE_DIR>().empty() || updatedConfig.get<BYPASS_UMD_CACHING>();
        auto graphDesc = _zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, bypassCache);
        _logger.debug("compile end");

        OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
        NetworkMetadata networkMetadata = _zeGraphExt->getNetworkMeta(graphDesc);

        if (isInitMetadata(networkMetadata)) {
            networkMetadata.name = model->get_friendly_name() + "_init";
            initNetworkMetadata.push_back(std::move(networkMetadata));
            initGraphDescriptors.push_back(graphDesc);
        } else {
            networkMetadata.name = model->get_friendly_name() + "_main";
            mainNetworkMetadata = std::move(networkMetadata);
            mainGraphHandle = graphDesc;
            serializedIR = SerializedIR();
            // By convention, the main schedule is the last result produced by the compiler
            break;
        }
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        auto compile_model_mem_end = get_peak_memory_usage();
        _logger.debug("Start of compilation memory usage: Peak %lld KB", compile_model_mem_start);
        _logger.debug("End of compilation memory usage: Peak %lld KB", compile_model_mem_end);
        // Note: Following log is parsed by CI. Take care when modifying it.
        _logger.info("Compilation memory usage: Peak %lld KB", compile_model_mem_end - compile_model_mem_start);
    }

    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             mainGraphHandle,
                                             std::move(mainNetworkMetadata),
                                             /* mainBlob = */ std::nullopt,
                                             initGraphDescriptors,
                                             std::move(initNetworkMetadata),
                                             /* initBlobs = */ std::nullopt,
                                             std::move(model),
                                             updatedConfig);
}

ov::SupportedOpsMap DriverCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(query_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "query");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    const auto isOptionValueSupportedByCompiler = [this](const std::string& optionName,
                                                         const std::optional<std::string>& optionValue) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionValueSupportedByCompiler);

    FilteredConfig updatedConfig = config;
    if (config.isAvailable(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update({{ov::intel_npu::model_serializer_version.name(),
                               MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion)}});
    }
    const auto isOptionSupportedByCompiler = std::bind(&DriverCompilerAdapter::isCompilerOptionSupported,
                                                       this,
                                                       std::cref(updatedConfig),
                                                       std::cref(compilerVersion),
                                                       std::placeholders::_1);

    std::string buildFlags;
    buildFlags += compiler_utils::serializeConfig(updatedConfig, compilerVersion, isOptionSupportedByCompiler);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    ov::SupportedOpsMap result;
    const std::string deviceName = "NPU";

    try {
        const auto supportedLayers = _zeGraphExt->queryGraph(std::move(serializedIR), buildFlags);
        for (auto&& layerName : supportedLayers) {
            result.emplace(layerName, deviceName);
        }
        _logger.info("For given model, there are %d supported layers", supportedLayers.size());
    } catch (std::exception& e) {
        OPENVINO_THROW("Fail in calling querynetwork : ", e.what());
    }

    _logger.debug("query end");
    return result;
}

uint32_t DriverCompilerAdapter::get_version() const {
    return _zeroInitStruct->getCompilerVersion();
}

std::optional<std::vector<std::string>> DriverCompilerAdapter::get_supported_options() const {
    std::optional<std::string> compilerOptionsStr;
    compilerOptionsStr = _zeGraphExt->getCompilerSupportedOptions();

    if (!compilerOptionsStr.has_value()) {
        return std::nullopt;
    }

    // vectorize string
    std::istringstream suppstream(compilerOptionsStr.value());
    std::vector<std::string> compilerOpts;
    std::string option;
    while (suppstream >> option) {
        compilerOpts.push_back(option);
    }
    return compilerOpts;
}

bool DriverCompilerAdapter::is_option_supported(std::string optName, std::optional<std::string> optValue) const {
    auto isOptionSupported = _zeGraphExt->isOptionSupported(std::move(optName), std::move(optValue));
    return isOptionSupported.value_or(false);
}

bool DriverCompilerAdapter::isCompilerOptionSupported(const FilteredConfig& config,
                                                      const ze_graph_compiler_version_info_t& compilerVersion,
                                                      const std::string& optionName) const {
    if (!config.hasOpt(optionName)) {
        return false;
    }

    const std::optional<bool> isSupported = _zeGraphExt->isOptionSupported(optionName);
    if (isSupported.has_value()) {
        return isSupported.value();
    }

    uint32_t compilerOptSupportValue = config.getOpt(optionName).compilerSupportVersion();
    uint32_t majorCompilerOptSupportValue = ZE_MAJOR_VERSION(compilerOptSupportValue);
    uint32_t minorCompilerOptSupportValue = ZE_MINOR_VERSION(compilerOptSupportValue);
    return (compilerVersion.major > majorCompilerOptSupportValue) ||
           ((compilerVersion.major == majorCompilerOptSupportValue) &&
            (compilerVersion.minor >= minorCompilerOptSupportValue));
}

bool DriverCompilerAdapter::validate_compatibility_descriptor(const std::string& compatibilityDescriptor,
                                                              uint32_t deviceId, int64_t numTiles, int64_t stepping) const {
    if (!_zeGraphExt->isCompatibilityDescriptorSupported()) {
        OPENVINO_THROW("");
    }
    return _zeGraphExt->validateCompatibilityDescriptor(compatibilityDescriptor);
}

}  // namespace intel_npu
