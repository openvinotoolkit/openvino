// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "compiler_schedules_sections.hpp"
#include "graph.hpp"
#include "intel_npu/common/encrypted_schedules_flag_section.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "mem_usage.hpp"
#include "model_serializer.hpp"
#include "openvino/core/model.hpp"
#include "weightless_graph.hpp"
#include "weightless_utils.hpp"

namespace intel_npu {

namespace {
constexpr OptionSupportCache::CacheKey driverOptionSupportKey =
    static_cast<OptionSupportCache::CacheKey>(ov::intel_npu::CompilerType::DRIVER);
}

namespace {

struct PropertySupportInfo {
    std::string name;
    uint32_t version;
};

bool isVersionSupportedByCompiler(uint32_t version, const ze_graph_compiler_version_info_t& compilerVersion) {
    const auto major = ONEAPI_VERSION_MAJOR(version);
    const auto minor = ONEAPI_VERSION_MINOR(version);
    return (compilerVersion.major > major) || ((compilerVersion.major == major) && (compilerVersion.minor >= minor));
}

const std::vector<PropertySupportInfo> _supportedPropertiesWithVersions = {
    {ov::compilation_num_threads.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::enable_profiling.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::hint::execution_mode.name(), ONEAPI_MAKE_VERSION(5, 6)},
    {ov::hint::inference_precision.name(), ONEAPI_MAKE_VERSION(5, 4)},
    {ov::hint::performance_mode.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::log::level.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::intel_npu::batch_compiler_mode_settings.name(), ONEAPI_MAKE_VERSION(7, 4)},
    {ov::intel_npu::batch_mode.name(), ONEAPI_MAKE_VERSION(5, 5)},
    {ov::intel_npu::backend_compilation_params.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::intel_npu::compilation_mode.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::intel_npu::compilation_mode_params.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::intel_npu::compiler_dynamic_quantization.name(), ONEAPI_MAKE_VERSION(7, 1)},
    {ov::intel_npu::dma_engines.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::intel_npu::dynamic_shape_to_static.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::intel_npu::max_tiles.name(), ONEAPI_MAKE_VERSION(5, 3)},
    {ov::intel_npu::platform.name(), ONEAPI_MAKE_VERSION(0, 0)},
    {ov::intel_npu::qdq_optimization.name(), ONEAPI_MAKE_VERSION(7, 20)},
    {ov::intel_npu::tiles.name(), ONEAPI_MAKE_VERSION(5, 4)},
    {ov::intel_npu::turbo.name(), ONEAPI_MAKE_VERSION(7, 21)},
    {ov::intel_npu::stepping.name(), ONEAPI_MAKE_VERSION(5, 3)},
};

}  // namespace

DriverCompilerAdapter::DriverCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                                             const std::shared_ptr<OptionSupportCache>& optionSupportCache)
    : _zeroInitStruct(zeroInitStruct),
      _optionSupportCache(optionSupportCache),
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
                                                       const FilteredConfig& config,
                                                       const std::shared_ptr<BlobWriter>& blobWriter) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compile");
    OPENVINO_ASSERT(blobWriter, "Requested compilation without providing a blob writer object");

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
    const auto isOptionSupportedByCompiler = [this](const std::string& optionName) {
        return is_option_supported(optionName);
    };

    _logger.debug("build flags");
    buildFlags += compiler_utils::serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += compiler_utils::serializeConfig(updatedConfig, compilerVersion, isOptionSupportedByCompiler);

    _logger.debug("compileIR Build flags : %s", buildFlags.c_str());

    _logger.debug("compile start");
    // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
    const bool bypassCache = !updatedConfig.get<CACHE_DIR>().empty() || updatedConfig.get<BYPASS_UMD_CACHING>();
    // If blob encryption is requested, enable secure compilation in the driver
    const bool secureCompile = updatedConfig.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
                               updatedConfig.get<CACHE_ENCRYPTION_CALLBACKS>().encrypt != nullptr;
    auto graphDesc = _zeGraphExt->getGraphDescriptor(std::move(serializedIR), buildFlags, bypassCache, secureCompile);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(graphDesc);
    networkMeta.name = model->get_friendly_name();

    auto graph = std::make_shared<Graph>(_zeGraphExt,
                                         _zeroInitStruct,
                                         graphDesc,
                                         std::move(networkMeta),
                                         /* blob = */ std::nullopt,
                                         updatedConfig,
                                         get_compatibility_descriptor(graphDesc._handle));

    if (secureCompile) {
        blobWriter->register_section(std::make_shared<EncryptedSchedulesFlagSection>(true));
    }

    // Tell the blob writer to store the main schedule in the blob at export time
    blobWriter->register_section(std::make_shared<ELFMainScheduleSection>(
        graph,
        secureCompile ? std::make_optional<>(updatedConfig.get<CACHE_ENCRYPTION_CALLBACKS>()) : std::nullopt,
        _logger.level()));

    return graph;
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compileWS(std::shared_ptr<ov::Model>&& model,
                                                         const FilteredConfig& config,
                                                         const std::shared_ptr<BlobWriter>& blobWriter) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compileWS");
    OPENVINO_ASSERT(blobWriter, "Requested compilation without providing a blob writer object");

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

    const auto isOptionSupportedByCompiler = [this](const std::string& optionName) {
        return is_option_supported(optionName);
    };

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

    auto weightlessGraph = std::make_shared<WeightlessGraph>(_zeGraphExt,
                                                             _zeroInitStruct,
                                                             mainGraphHandle,
                                                             std::move(mainNetworkMetadata),
                                                             /* mainBlob = */ std::nullopt,
                                                             initGraphDescriptors,
                                                             std::move(initNetworkMetadata),
                                                             /* initBlobs = */ std::nullopt,
                                                             std::move(model),
                                                             updatedConfig,
                                                             /* persistentBlob = */ false,
                                                             get_compatibility_descriptor(mainGraphHandle._handle));

    std::optional<ov::EncryptionCallbacks> encryptionCallbacks = std::nullopt;
    if (updatedConfig.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
        updatedConfig.get<CACHE_ENCRYPTION_CALLBACKS>().encrypt != nullptr) {
        encryptionCallbacks = updatedConfig.get<CACHE_ENCRYPTION_CALLBACKS>();
        blobWriter->register_section(std::make_shared<EncryptedSchedulesFlagSection>(true));
    }

    // At export time, all schedules (main + inits) shall be stored in the blob.
    blobWriter->register_section(
        std::make_shared<ELFMainScheduleSection>(weightlessGraph, encryptionCallbacks, _logger.level()));
    blobWriter->register_section(
        std::make_shared<ELFInitSchedulesSection>(weightlessGraph, encryptionCallbacks, _logger.level()));

    return weightlessGraph;
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
    const auto isOptionSupportedByCompiler = [this](const std::string& optionName) {
        return is_option_supported(optionName);
    };

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

std::vector<std::string> DriverCompilerAdapter::get_supported_options() const {
    std::optional<std::string> compilerOptionsStr;
    compilerOptionsStr = _zeGraphExt->getCompilerSupportedOptions();

    std::vector<std::string> compilerOpts;

    if (compilerOptionsStr.has_value()) {
        if (compilerOptionsStr->empty()) {
            _logger.info("get_supported_options returned no options; returning an empty supported options vector.");
            return {};
        }

        // vectorize string
        std::istringstream suppstream(compilerOptionsStr.value());
        std::string option;
        while (suppstream >> option) {
            compilerOpts.push_back(option);
        }

        if (_optionSupportCache) {
            _optionSupportCache->setSupportedOptions(driverOptionSupportKey, compilerOpts);
        }
        return compilerOpts;
    }

    // legacy path
    const auto& compilerVersion = _compilerProperties.compilerVersion;
    for (const auto& prop : _supportedPropertiesWithVersions) {
        if (isVersionSupportedByCompiler(prop.version, compilerVersion)) {
            compilerOpts.push_back(prop.name);
        }
    }

    if (compilerOpts.empty()) {
        return {};
    }

    if (_optionSupportCache) {
        _optionSupportCache->setSupportedOptions(driverOptionSupportKey, compilerOpts);
    }
    return compilerOpts;
}

bool DriverCompilerAdapter::is_option_supported(const std::string& optName,
                                                const std::optional<std::string>& optValue) const {
    bool optionSupportCache = _optionSupportCache && !optValue.has_value();
    if (optionSupportCache) {
        const auto cachedSupport = _optionSupportCache->isOptionSupported(driverOptionSupportKey, optName);
        if (cachedSupport.has_value()) {
            return cachedSupport.value();
        }
    }

    auto isOptionSupported = _zeGraphExt->isOptionSupported(optName, optValue);
    if (isOptionSupported.has_value()) {
        const bool supported = isOptionSupported.value();
        if (optionSupportCache) {
            _optionSupportCache->addSupportedOption(driverOptionSupportKey, optName, supported);
        }

        return supported;
    }

    // legacy path
    const auto& compilerVersion = _compilerProperties.compilerVersion;
    for (const auto& prop : _supportedPropertiesWithVersions) {
        if (prop.name == optName) {
            const bool supported = isVersionSupportedByCompiler(prop.version, compilerVersion);
            if (_optionSupportCache) {
                _optionSupportCache->addSupportedOption(driverOptionSupportKey, optName, supported);
            }
            return supported;
        }
    }
    return false;
}

std::optional<std::string> DriverCompilerAdapter::get_compatibility_descriptor(ze_graph_handle_t graphHandle) const {
    return _zeGraphExt->getCompatibilityDescriptor(graphHandle);
}

}  // namespace intel_npu
