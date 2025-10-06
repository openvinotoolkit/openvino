// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include <chrono>
#include <string_view>

#include "graph.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "ir_serializer.hpp"
#include "mem_usage.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "weightless_graph.hpp"

namespace {

bool isInitMetadata(const intel_npu::NetworkMetadata& networkMetadata) {
    if (networkMetadata.inputs.size() == 0) {
        return false;
    }
    return networkMetadata.inputs.at(0).isInitInputWeights;
}

/**
 * @brief Stores the information within the "WeightlessCacheAttribute" as runtime fields that persist upon
 * serialization.
 * @details Constant nodes (weights) may contain as medatadata the "WeightlessCacheAttribute", that is information
 * regarding the offset of the weights within the binary file, as well as the original size and precision. This
 * information is required within the "weights separation" flow, therefore this function is here to store it.
 * @note Not calling this function in the weights separation flow would lead to this information being lost upon
 * serialization. The "WeightlessCacheAttribute" information that is populated upon de-serialization would represent
 * metadata corresponding to the serialized stream, not the original weights file. Therefore the compiler would be
 * misinformed and lookups of weights offsets could fail.
 *
 * @param model Both source and target.
 */
void storeWeightlessCacheAttribute(const std::shared_ptr<ov::Model>& model) {
    size_t constantId = 0;
    for (auto&& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            ov::RTMap& runtimeInfoMap = node->get_rt_info();
            const auto& weightlessCacheAttrIt =
                runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());

            const std::string constantIdString = std::to_string(constantId++);
            if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
                auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();
                model->set_rt_info(weightlessCacheAttr.bin_offset, "ws_bin_offset_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_size, "ws_original_size_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_dtype, "ws_original_dtype_" + constantIdString);
            }
        }
    }
}

// clang-format off
#include "windows.h"
#include "psapi.h"
// clang-format on

void printMemoryValue() {  // Note: this value is in KB!
    PROCESS_MEMORY_COUNTERS_EX2 pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    std::cout << "VM: " << pmc.WorkingSetSize / 1024 << " PM: " << pmc.PrivateWorkingSetSize / 1024 << " KB"
              << std::endl;
}

}  // namespace

namespace intel_npu {

DriverCompilerAdapter::DriverCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("DriverCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize DriverCompilerAdapter start");

    uint32_t graphExtVersion = _zeroInitStruct->getGraphDdiTable().version();

    _compilerProperties = _zeroInitStruct->getCompilerProperties();

    _logger.info("DriverCompilerAdapter creating adapter using graphExtVersion");

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

    std::cout << "Before serialization" << std::endl;
    printMemoryValue();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto serializedIR = driver_compiler_utils::serializeIR(
        model,
        compilerVersion,
        maxOpsetVersion,
        config.isAvailable(ov::intel_npu::use_base_model_serializer.name()) ? config.get<USE_BASE_MODEL_SERIALIZER>()
                                                                            : false);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time to serialize: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[µs]" << std::endl;
    std::cout << "After serialization" << std::endl;
    printMemoryValue();

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    _logger.debug("build flags");
    buildFlags += driver_compiler_utils::serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += driver_compiler_utils::serializeConfig(config, compilerVersion, is_option_supported("NPU_TURBO"));

    _logger.debug("compileIR Build flags : %s", buildFlags.c_str());

    // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
    uint32_t flags = ZE_GRAPH_FLAG_NONE;
    const auto set_cache_dir = config.get<CACHE_DIR>();
    if (!set_cache_dir.empty() || config.get<BYPASS_UMD_CACHING>()) {
        flags = flags | ZE_GRAPH_FLAG_DISABLE_CACHING;
    }

    _logger.debug("compile start");
    auto graphDesc = _zeGraphExt->getGraphDescriptor(std::move(serializedIR), buildFlags, flags);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(graphDesc);
    networkMeta.name = model->get_friendly_name();

    return std::make_shared<Graph>(_zeGraphExt,
                                   _zeroInitStruct,
                                   graphDesc,
                                   std::move(networkMeta),
                                   /* blob = */ std::nullopt,
                                   config);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compileWS(const std::shared_ptr<ov::Model>& model,
                                                         const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compileWS");

    storeWeightlessCacheAttribute(model);

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
    std::cout << "Before serialization" << std::endl;
    printMemoryValue();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto serializedIR = driver_compiler_utils::serializeIR(
        model,
        compilerVersion,
        maxOpsetVersion,
        config.isAvailable(ov::intel_npu::use_base_model_serializer.name()) ? config.get<USE_BASE_MODEL_SERIALIZER>()
                                                                            : false);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time to serialize: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[µs]" << std::endl;
    std::cout << "After serialization" << std::endl;
    printMemoryValue();

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    const std::string serializedIOInfo = driver_compiler_utils::serializeIOInfo(model, useIndices);
    const FilteredConfig* plgConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (plgConfig == nullptr) {
        OPENVINO_THROW("config is not FilteredConfig");
    }
    FilteredConfig updatedConfig = *plgConfig;

    // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
    uint32_t flags = ZE_GRAPH_FLAG_NONE;
    const auto set_cache_dir = config.get<CACHE_DIR>();
    if (!set_cache_dir.empty() || config.get<BYPASS_UMD_CACHING>()) {
        flags = flags | ZE_GRAPH_FLAG_DISABLE_CACHING;
    }

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
    while (true) {
        _logger.debug("compileWS iteration %d", callNumber);
        updatedConfig.update({{ov::intel_npu::ws_compile_call_number.name(), std::to_string(callNumber++)}});

        _logger.debug("build flags");
        buildFlags = serializedIOInfo;
        buildFlags += " ";
        buildFlags += driver_compiler_utils::serializeConfig(updatedConfig, compilerVersion);

        _logger.debug("compile start");
        auto graphDesc = _zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, flags);
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
                                             model,
                                             config);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::parse(
    ov::Tensor mainBlob,
    const FilteredConfig& config,
    std::optional<std::vector<ov::Tensor>> initBlobs,
    const std::optional<std::shared_ptr<const ov::Model>>& model) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "parse");

    _logger.debug("parse start");
    auto mainGraphDesc = _zeGraphExt->getGraphDescriptor(mainBlob.data(), mainBlob.get_byte_size());
    _logger.debug("parse end");

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(mainGraphDesc);

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
                                       blobIsPersistent);
    }

    // The presence of init schedules means weights separation has been enabled at compilation time. Use a specific
    // "Graph" object as wrapper over all L0 handles.
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initMetadata;

    for (const auto& initBlob : initBlobs.value()) {
        auto initGraphDesc = _zeGraphExt->getGraphDescriptor(initBlob.data(), initBlob.get_byte_size());

        initGraphDescriptors.push_back(initGraphDesc);
        initMetadata.push_back(_zeGraphExt->getNetworkMeta(initGraphDesc));
    }

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
                                             blobIsPersistent);
}

ov::SupportedOpsMap DriverCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(query_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "query");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    auto serializedIR = driver_compiler_utils::serializeIR(
        model,
        compilerVersion,
        maxOpsetVersion,
        config.isAvailable(ov::intel_npu::use_base_model_serializer.name()) ? config.get<USE_BASE_MODEL_SERIALIZER>()
                                                                            : false);

    std::string buildFlags;
    buildFlags += driver_compiler_utils::serializeConfig(config, compilerVersion);
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
    std::string compilerOptionsStr;
    compilerOptionsStr = _zeGraphExt->getCompilerSupportedOptions();
    // vectorize string
    std::istringstream suppstream(compilerOptionsStr);
    std::vector<std::string> compilerOpts;
    std::string option;
    while (suppstream >> option) {
        compilerOpts.push_back(option);
    }
    return compilerOpts;
}

bool DriverCompilerAdapter::is_option_supported(std::string optname) const {
    return _zeGraphExt->isOptionSupported(std::move(optname));
}

}  // namespace intel_npu
