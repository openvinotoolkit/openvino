// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_impl.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>

#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/profiling.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/vcl/vcl_allocator.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "model_serializer.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "vcl_version_utils.hpp"
#include "weightless_utils.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

using vcl_version_utils::checkVclVersion;
using vcl_version_utils::getUsedVclVersion;
using vcl_version_utils::UsedVersion;

static inline std::string getLatestVCLLog(const VCLApi& api, vcl_log_handle_t logHandle) {
    Logger _logger("VCLAPI", Logger::global().level());
    _logger.debug("getLatestVCLLog start");

    vcl_version_info_t compilerVersion;
    vcl_version_info_t profilingVersion;
    vcl_result_t ret = api.vclGetVersion(&compilerVersion, &profilingVersion);

    if (ret != VCL_RESULT_SUCCESS || compilerVersion.major < 3) {
        _logger.warning("Failed to get VCL version: 0x%x", ret);
        return "Can not get VCL log, VCL version is too old!";
    }

    // Get log size
    size_t size = 0;
    // Null graph handle to get error log
    ret = api.vclLogHandleGetString(logHandle, &size, nullptr);
    if (VCL_RESULT_SUCCESS != ret) {
        return "Failed to get size of latest VCL log";
    }

    if (size <= 0) {
        return "No error stored in VCL when error detected";
    }

    // Get log content
    std::string logContent{};
    logContent.resize(size);
    ret = api.vclLogHandleGetString(logHandle, &size, const_cast<char*>(logContent.data()));
    if (VCL_RESULT_SUCCESS != ret) {
        return "Size of latest error log > 0, failed to get content";
    }
    _logger.debug("getLatestBuildError end");
    return logContent;
}

static std::optional<std::string> getVCLCompatibilityString(const VCLApi& api,
                                                            vcl_executable_handle_t executable,
                                                            vcl_log_handle_t logHandle) {
    uint64_t compatibilityStringSize = 0;
    auto result = api.vclExecutableGetCompatibilityString(executable, nullptr, &compatibilityStringSize);
    if (result == VCL_RESULT_ERROR_UNSUPPORTED_FEATURE) {
        return std::nullopt;
    }
    if (result != VCL_RESULT_SUCCESS || compatibilityStringSize == 0) {
        OPENVINO_THROW("Failed to get compatibility string size. vclExecutableGetCompatibilityString result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(api, logHandle));
    }

    if (compatibilityStringSize > std::numeric_limits<size_t>::max()) {
        OPENVINO_THROW("Compatibility string size is too large to allocate a local buffer");
    }
    std::string compatibilityString(static_cast<size_t>(compatibilityStringSize), '\0');
    result = api.vclExecutableGetCompatibilityString(executable, compatibilityString.data(), &compatibilityStringSize);
    if (result != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to get compatibility string. vclExecutableGetCompatibilityString result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(api, logHandle));
    }
    if (compatibilityStringSize > compatibilityString.size()) {
        OPENVINO_THROW("Returned compatibility string size exceeds the allocated buffer size");
    }

    const size_t outSize = static_cast<size_t>(compatibilityStringSize);
    compatibilityString.resize(outSize);
    if (outSize > 0 && compatibilityString[outSize - 1] == '\0') {
        compatibilityString.resize(outSize - 1);
    }
    return compatibilityString;
}

#define THROW_ON_FAIL_FOR_VCL(step, ret, logHandle)            \
    {                                                          \
        vcl_result_t result = ret;                             \
        if (result != VCL_RESULT_SUCCESS) {                    \
            OPENVINO_THROW("Failed to call VCL API : ",        \
                           step,                               \
                           " result: 0x",                      \
                           std::hex,                           \
                           result,                             \
                           " - ",                              \
                           getLatestVCLLog(*_api, logHandle)); \
        }                                                      \
    }

VCLCompilerImpl::VCLCompilerImpl(std::shared_ptr<const VCLApi> api,
                                 const std::optional<IDevice::DeviceProperties>& deviceProperties)
    : _api(std::move(api)),
      _logHandle(nullptr),
      _logger("VCLCompilerImpl", Logger::global().level()) {
    _logger.debug("VCLCompilerImpl constructor start");

    OPENVINO_ASSERT(_api != nullptr, "VCLCompilerImpl requires a non-null VCL API table");

    // Initialize the VCL API
    THROW_ON_FAIL_FOR_VCL("vclGetVersion", _api->vclGetVersion(&_vclVersion, &_vclProfilingVersion), nullptr);
    _logger.info("Plugin VCL API Version: %d.%d", VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR);
    _logger.info("Plugin VCL Profiling API Version: %d.%d", VCL_PROFILING_VERSION_MAJOR, VCL_PROFILING_VERSION_MINOR);
    _logger.info("Lib VCL Compiler Version: %d.%d", _vclVersion.major, _vclVersion.minor);
    _logger.info("Lib VCL Profiling Version: %d.%d", _vclProfilingVersion.major, _vclProfilingVersion.minor);
    if (VCL_COMPILER_VERSION_MAJOR < _vclVersion.major ||
        (VCL_COMPILER_VERSION_MAJOR == _vclVersion.major && VCL_COMPILER_VERSION_MINOR < _vclVersion.minor)) {
        _logger.warning("inside supported VCL version is lower than loaded VCL api:\n plugin was built with VCL %d.%d, "
                        "\n      but loaded VCL is %d.%d.\n"
                        "Will downgrade to use the plugin vcl compiler",
                        VCL_COMPILER_VERSION_MAJOR,
                        VCL_COMPILER_VERSION_MINOR,
                        _vclVersion.major,
                        _vclVersion.minor);
    } else {
        _logger.info("Use Lib VCL version to create compiler");
    }

    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version = _vclVersion;
    compilerDesc.debugLevel = static_cast<__vcl_log_level_t>(static_cast<int>(Logger::global().level()) + 1);

    vcl_device_desc_t vclDeviceDesc = {};
    if (deviceProperties.has_value()) {
        constexpr auto invalidRevision = std::numeric_limits<uint16_t>::max();
        const auto revision = deviceProperties->subdeviceId >= invalidRevision
                                  ? invalidRevision
                                  : static_cast<uint16_t>(deviceProperties->subdeviceId);

        if (revision == invalidRevision) {
            _logger.warning("Device subdeviceId %u does not fit into VCL revision field; using invalid revision "
                            "sentinel instead",
                            deviceProperties->subdeviceId);
        }

        _logger.info("Device description is provided, using deviceID: 0x%X, subdeviceID: %u, maxTiles: %u",
                     deviceProperties->deviceId,
                     deviceProperties->subdeviceId,
                     deviceProperties->numSlices);
        vclDeviceDesc = {sizeof(vcl_device_desc_t), deviceProperties->deviceId, revision, deviceProperties->numSlices};
    } else {
        // This information cannot be determined during the initialization phase; set device desc default value, the
        // related info will be processed in compile phase if passed by user.
        _logger.info("Device description is not provided, using default values");
        uint32_t defaultTileCount = std::numeric_limits<uint32_t>::max();
        vclDeviceDesc = {sizeof(vcl_device_desc_t), 0x00, std::numeric_limits<uint16_t>::max(), defaultTileCount};
    }

    THROW_ON_FAIL_FOR_VCL("vclCompilerCreate",
                          _api->vclCompilerCreate(&compilerDesc, &vclDeviceDesc, &_compilerHandle, &_logHandle),
                          nullptr);
    THROW_ON_FAIL_FOR_VCL("vclCompilerGetProperties",
                          _api->vclCompilerGetProperties(_compilerHandle, &_compilerProperties),
                          _logHandle);
    _logger.info("VCL Compiler created successfully");
    _logger.info("VCL Compiler Properties: ID: %s, Version: %d.%d, Supported Opsets: %u",
                 _compilerProperties.id,
                 _compilerProperties.version.major,
                 _compilerProperties.version.minor,
                 _compilerProperties.supportedOpsets);
}

VCLCompilerImpl::~VCLCompilerImpl() {
    if (_compilerHandle) {
        vcl_result_t result = _api->vclCompilerDestroy(_compilerHandle);
        _compilerHandle = nullptr;
        if (result != VCL_RESULT_SUCCESS) {
            _logger.warning("Failed to destroy VCL compiler: result 0x%x - %s",
                            result,
                            getLatestVCLLog(*_api, _logHandle).c_str());
        }
    }

    if (_logHandle) {
        _logHandle = nullptr;  // Log handle is released automatically with the compiler
    }
    _logger.info("VCL Compiler destroyed successfully");
}

std::shared_ptr<void> VCLCompilerImpl::getLinkedLibrary() const {
    return _api->getLibrary();
}

std::pair<ov::Tensor, std::optional<std::string>> VCLCompilerImpl::compile(
    const std::shared_ptr<const ov::Model>& model,
    const FilteredConfig& config) const {
    return compile(model, config, false);
}

std::pair<ov::Tensor, std::optional<std::string>> VCLCompilerImpl::compile(
    const std::shared_ptr<const ov::Model>& model,
    const FilteredConfig& config,
    const bool storeWeightlessCacheAttributeFlag) const {
    _logger.debug("compile start");

    /// Check the linked vcl version whether supported in plugin
    UsedVersion usedVersion =
        getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion.major, _vclVersion.minor);
    _logger.debug("the finally used compiler vcl version is %d.%d", usedVersion.Major, usedVersion.Minor);
    checkVclVersion(usedVersion,
                    _vclVersion.major,
                    _vclVersion.minor,
                    VCL_COMPILER_VERSION_MAJOR,
                    VCL_COMPILER_VERSION_MINOR);

    const auto maxOpsetVersion = _compilerProperties.supportedOpsets;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ze_graph_compiler_version_info_t compilerVersion;
    compilerVersion.major = _compilerProperties.version.major;
    compilerVersion.minor = _compilerProperties.version.minor;

    const auto isOptionValueSupportedByCompiler = [this](const std::string& optionName,
                                                         const std::optional<std::string>& optionValue) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionValueSupportedByCompiler,
                                                    false,
                                                    storeWeightlessCacheAttributeFlag);
    FilteredConfig updatedConfig = config;
    if (config.isAvailable(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update({{ov::intel_npu::model_serializer_version.name(),
                               MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion)}});
    }

    std::string buildFlags;
    const auto isOptionSupportedByCompiler = [this](const std::string& optionName) {
        return is_option_supported(optionName);
    };

    _logger.debug("create build flags");
    buildFlags += compiler_utils::serializeIOInfo(model, true);
    buildFlags += " ";
    buildFlags += compiler_utils::serializeConfig(updatedConfig, compilerVersion, isOptionSupportedByCompiler);

    _logger.debug("final build flags to compiler: %s", buildFlags.c_str());

    vcl_executable_desc_t exeDesc = {serializedIR.buffer.get(),
                                     serializedIR.size,
                                     buildFlags.c_str(),
                                     buildFlags.size()};
    // Support only the lastest VCL api
    auto allocator = std::make_shared<vcl_allocator_2>();
    uint8_t* blob = nullptr;
    uint64_t blobSize = 0;
    vcl_executable_handle_t executable = nullptr;

    auto result =
        _api->vclAllocatedExecutableCreate4(_compilerHandle, exeDesc, allocator.get(), &blob, &blobSize, &executable);
    if (result != VCL_RESULT_SUCCESS) {
        // Check if allocations were performed before throwing exception
        auto tracked_allocations = allocator->m_info;
        for (const auto& [buffer, size] : tracked_allocations) {
            allocator->deallocate(allocator.get(), buffer);
        }
        if (executable != nullptr) {
            _api->vclExecutableDestroy(executable);
        }
        OPENVINO_THROW("Compilation failed. vclAllocatedExecutableCreate4 result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(*_api, _logHandle));
    }
    OPENVINO_ASSERT(executable != nullptr, "Failed to create VCL executable, executable handle is null");
    OPENVINO_ASSERT(blobSize != 0 && blob != nullptr,
                    "Failed to create VCL executable, the blob size is zero or the blob is null");

    // Retrieve the real allocated size for the blob from the allocator
    auto it = std::find_if(allocator->m_info.begin(),
                           allocator->m_info.end(),
                           [blob](const std::pair<uint8_t*, size_t>& item) {
                               return item.first == blob;
                           });

    OPENVINO_ASSERT(it != allocator->m_info.end(), "Failed to find the allocated blob in the allocator records");
    size_t alignedBlobSize = it->second;

    // The allocated size from VCL will be equal or smaller than the allocated size in allocator
    _logger.debug("Blob size from VCL: %zu ptr %p", static_cast<size_t>(blobSize), static_cast<void*>(blob));
    _logger.debug("Allocated vector size: %zu ptr: %p", alignedBlobSize, static_cast<void*>(blob));

    ov::Tensor alignedBlob = make_tensor_from_aligned_addr(blob, alignedBlobSize, allocator);
    allocator->m_info.erase(it);

    std::optional<std::string> compatibilityString;
    try {
        compatibilityString = getVCLCompatibilityString(*_api, executable, _logHandle);
    } catch (...) {
        _api->vclExecutableDestroy(executable);
        throw;
    }
    if (!compatibilityString.has_value()) {
        // Some compilation modes (e.g. HostCompile_Interpreter) do not produce a compatibility descriptor.
        _logger.info("vclExecutableGetCompatibilityString is not supported for this executable (0x%x); "
                     "compatibility string will be absent",
                     uint32_t(VCL_RESULT_ERROR_UNSUPPORTED_FEATURE));
    } else {
        _logger.debug("Compatibility string from VCL: %s", compatibilityString->c_str());
    }

    result = _api->vclExecutableDestroy(executable);
    if (result != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to destroy VCL executable. vclExecutableDestroy result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(*_api, _logHandle));
    }

    return std::make_pair<ov::Tensor, std::optional<std::string>>(std::move(alignedBlob),
                                                                  std::move(compatibilityString));
}

std::pair<std::vector<ov::Tensor>, std::optional<std::string>> VCLCompilerImpl::compileWsOneShot(
    const std::shared_ptr<ov::Model>& model,
    const FilteredConfig& config) const {
    _logger.debug("compileWsOneShot start");

    /// Check the linked vcl version whether supported in plugin
    UsedVersion usedVersion =
        getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion.major, _vclVersion.minor);
    _logger.debug("the finally used compiler vcl version is %d.%d", usedVersion.Major, usedVersion.Minor);
    checkVclVersion(usedVersion,
                    _vclVersion.major,
                    _vclVersion.minor,
                    VCL_COMPILER_VERSION_MAJOR,
                    VCL_COMPILER_VERSION_MINOR);

    const auto maxOpsetVersion = _compilerProperties.supportedOpsets;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ze_graph_compiler_version_info_t compilerVersion;
    compilerVersion.major = _compilerProperties.version.major;
    compilerVersion.minor = _compilerProperties.version.minor;

    const auto isOptionValueSupportedByCompiler = [this](const std::string& optionName,
                                                         const std::optional<std::string>& optionValue) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionValueSupportedByCompiler,
                                                    false,
                                                    true);
    FilteredConfig updatedConfig = config;
    if (config.isAvailable(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update({{ov::intel_npu::model_serializer_version.name(),
                               MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion)}});
    }

    std::string buildFlags;
    const auto isOptionSupportedByCompiler = [this](const std::string& optionName) {
        return is_option_supported(optionName);
    };

    _logger.debug("create build flags");
    buildFlags += compiler_utils::serializeIOInfo(model, true);
    buildFlags += " ";
    buildFlags += compiler_utils::serializeConfig(updatedConfig, compilerVersion, isOptionSupportedByCompiler);
    _logger.debug("final build flags to compiler: %s", buildFlags.c_str());

    vcl_executable_desc_t exeDesc = {serializedIR.buffer.get(),
                                     serializedIR.size,
                                     buildFlags.c_str(),
                                     buildFlags.size()};
    _logger.debug("compiler vcl version: %d.%d", _vclVersion.major, _vclVersion.minor);

    _logger.debug("Using vclAllocatedExecutableCreateWSOneShot2");
    auto allocator = std::make_shared<vcl_allocator_2>();
    vcl_executable_handle_t executable = nullptr;

    auto result = _api->vclAllocatedExecutableCreateWSOneShot2(_compilerHandle, exeDesc, allocator.get(), &executable);
    if (result != VCL_RESULT_SUCCESS) {
        if (executable != nullptr) {
            _api->vclExecutableDestroy(executable);
        }
        OPENVINO_THROW("Compilation failed. vclAllocatedExecutableCreateWSOneShot2 result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(*_api, _logHandle));
    }
    if (executable == nullptr) {
        OPENVINO_THROW("Failed to create VCL executable, executable handle is null");
    }

    if (allocator->m_info.size() == 0) {
        _api->vclExecutableDestroy(executable);
        OPENVINO_THROW("Failed to create VCL executable, blobCount is zero");
    }

    std::vector<ov::Tensor> initMainTensors;
    for (const auto& blob : allocator->m_info) {
        initMainTensors.emplace_back(make_tensor_from_aligned_addr(blob.first, blob.second, allocator));
    }
    // Clean up m_info, delegating actual physical frees strictly to the Tensor/Deleter from now on.
    allocator->m_info.clear();

    std::optional<std::string> compatibilityString;
    try {
        compatibilityString = getVCLCompatibilityString(*_api, executable, _logHandle);
    } catch (...) {
        _api->vclExecutableDestroy(executable);
        throw;
    }
    if (!compatibilityString.has_value()) {
        _logger.info("vclExecutableGetCompatibilityString is not supported for this executable (0x%x); "
                     "compatibility string will be absent",
                     uint32_t(VCL_RESULT_ERROR_UNSUPPORTED_FEATURE));
    } else {
        _logger.debug("Compatibility string from VCL: %s", compatibilityString->c_str());
    }

    result = _api->vclExecutableDestroy(executable);
    if (result != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to destroy executable. vclExecutableDestroy result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(*_api, _logHandle));
    }

    return std::make_pair(std::move(initMainTensors), std::move(compatibilityString));
}

std::pair<ov::Tensor, std::optional<std::string>> VCLCompilerImpl::compileWsIterative(
    const std::shared_ptr<ov::Model>& model,
    const FilteredConfig& config,
    size_t callNumber) const {
    _logger.debug("compileWsIterative start");
    FilteredConfig updatedConfig = config;
    updatedConfig.update({{ov::intel_npu::ws_compile_call_number.name(), std::to_string(callNumber)}});
    // Return the compatibility descriptor together with the compiled blob.
    return compile(model, updatedConfig, true);
}

std::vector<ov::ProfilingInfo> VCLCompilerImpl::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                         const std::vector<uint8_t>& network) const {
    _logger.debug("process_profiling_output start");

    vcl_profiling_handle_t profilingHandle;
    vcl_profiling_input_t profilingInput = {network.data(), network.size(), profData.data(), profData.size()};
    vcl_log_handle_t logHandle;
    THROW_ON_FAIL_FOR_VCL("vclProfilingCreate",
                          _api->vclProfilingCreate(&profilingInput, &profilingHandle, &logHandle),
                          nullptr);

    vcl_profiling_properties_t profProperties;
    THROW_ON_FAIL_FOR_VCL("vclProfilingGetProperties",
                          _api->vclProfilingGetProperties(profilingHandle, &profProperties),
                          logHandle);

    _logger.info("VCL Profiling Properties: Version: %d.%d",
                 profProperties.version.major,
                 profProperties.version.minor);

    // We only use layer level info
    vcl_profiling_request_type_t request = VCL_PROFILING_LAYER_LEVEL;

    vcl_profiling_output_t profOutput;
    profOutput.data = NULL;
    THROW_ON_FAIL_FOR_VCL("vclGetDecodedProfilingBuffer",
                          _api->vclGetDecodedProfilingBuffer(profilingHandle, request, &profOutput),
                          logHandle);
    if (profOutput.data == NULL) {
        OPENVINO_THROW("Failed to get VCL profiling output");
    }

    std::vector<ze_profiling_layer_info> layerInfo(profOutput.size / sizeof(ze_profiling_layer_info));
    if (profOutput.size > 0) {
        _logger.debug("VCL profiling output size: %d", profOutput.size);
        std::memcpy(layerInfo.data(), profOutput.data, profOutput.size);
    }

    THROW_ON_FAIL_FOR_VCL("vclProfilingDestroy", _api->vclProfilingDestroy(profilingHandle), logHandle);

    // Return processed profiling info
    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerInfo);
}

uint32_t VCLCompilerImpl::get_version() const {
    return ZE_MAKE_VERSION(_compilerProperties.version.major, _compilerProperties.version.minor);
}

ov::SupportedOpsMap VCLCompilerImpl::query(const std::shared_ptr<const ov::Model>& model,
                                           const FilteredConfig& config) const {
    _logger.debug("query start");

    /// Check the linked vcl version whether supported in plugin
    UsedVersion usedVersion =
        getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion.major, _vclVersion.minor);
    _logger.debug("the finally used vcl version is %d.%d", usedVersion.Major, usedVersion.Minor);

    const auto maxOpsetVersion = _compilerProperties.supportedOpsets;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ze_graph_compiler_version_info_t compilerVersion;
    compilerVersion.major = _compilerProperties.version.major;
    compilerVersion.minor = _compilerProperties.version.minor;
    FilteredConfig updatedConfig = config;
    const auto isOptionValueSupportedByCompiler = [this](const std::string& optionName,
                                                         const std::optional<std::string>& optionValue) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionValueSupportedByCompiler);
    if (config.isAvailable(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update({{ov::intel_npu::model_serializer_version.name(),
                               MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion)}});
    }

    std::string buildFlags;
    const auto isOptionSupportedByCompiler = [this](const std::string& optionName) {
        return is_option_supported(optionName);
    };
    buildFlags += compiler_utils::serializeConfig(updatedConfig, compilerVersion, isOptionSupportedByCompiler);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    vcl_query_handle_t queryHandle;
    vcl_query_desc_t queryDesc = {serializedIR.buffer.get(), serializedIR.size, buildFlags.c_str(), buildFlags.size()};
    THROW_ON_FAIL_FOR_VCL("vclQueryNetworkCreate",
                          _api->vclQueryNetworkCreate(_compilerHandle, queryDesc, &queryHandle),
                          _logHandle);

    uint64_t size = 0;
    THROW_ON_FAIL_FOR_VCL("vclQueryNetwork", _api->vclQueryNetwork(queryHandle, nullptr, &size), _logHandle);

    std::vector<char> supportedLayers(size);
    THROW_ON_FAIL_FOR_VCL("vclQueryNetwork",
                          _api->vclQueryNetwork(queryHandle, reinterpret_cast<uint8_t*>(supportedLayers.data()), &size),
                          _logHandle);

    THROW_ON_FAIL_FOR_VCL("vclQueryNetworkDestroy", _api->vclQueryNetworkDestroy(queryHandle), _logHandle);

    const std::string deviceName = "NPU";
    ov::SupportedOpsMap result;
    const auto parsedSupportedLayers = parseQueryResult(supportedLayers);
    for (auto&& layerName : parsedSupportedLayers) {
        result.emplace(layerName, deviceName);
    }
    _logger.info("For given model, there are %d supported layers", parsedSupportedLayers.size());

    return result;
}

std::vector<std::string> VCLCompilerImpl::get_supported_options() const {
    _logger.debug("get_supported_options start");
    size_t str_size = 0;
    THROW_ON_FAIL_FOR_VCL("vclGetCompilerSupportedOptions",
                          _api->vclGetCompilerSupportedOptions(_compilerHandle, nullptr, &str_size),
                          _logHandle);

    if (str_size == 0) {
        _logger.debug("Option list size 0!");
        _logger.info("get_supported_options returned no options; returning an empty supported options vector.");
        return {};
    }

    _logger.debug("obtain list");
    std::vector<char> options(str_size);
    THROW_ON_FAIL_FOR_VCL("vclGetCompilerSupportedOptions",
                          _api->vclGetCompilerSupportedOptions(_compilerHandle, options.data(), &str_size),
                          _logHandle);

    _logger.debug("Option list size %d, got option list", str_size);

    // VCL hands back a char buffer that may carry trailing NULs. Trimming and tokenising here keeps
    // that calling convention out of IVCLCompiler.
    size_t optionsSize = options.size();
    while (optionsSize > 0 && options[optionsSize - 1] == '\0') {
        --optionsSize;
    }
    if (optionsSize == 0) {
        _logger.info("get_supported_options returned no options; returning an empty supported options vector.");
        return {};
    }

    const std::string compilerOptionsStr(options.data(), optionsSize);
    _logger.debug("VCLCompilerImpl return supported_options: %s", compilerOptionsStr.c_str());
    std::istringstream suppstream(compilerOptionsStr);
    std::vector<std::string> compilerOpts;
    std::string option;
    while (suppstream >> option) {
        compilerOpts.push_back(option);
    }
    return compilerOpts;
}

bool VCLCompilerImpl::is_option_supported(const std::string& option, const std::optional<std::string>& optValue) const {
    try {
        const char* optname_ch = option.c_str();
        const char* optvalue_ch = optValue.has_value() ? optValue.value().c_str() : nullptr;
        _logger.debug("is_option_supported start for option: %s, value: %s",
                      optname_ch,
                      optvalue_ch ? optvalue_ch : "null");
        THROW_ON_FAIL_FOR_VCL("vclGetCompilerIsOptionSupported",
                              _api->vclGetCompilerIsOptionSupported(_compilerHandle, optname_ch, optvalue_ch),
                              _logHandle);
        return true;
    } catch (const std::exception& e) {
        // The API is only supported in new version, just add log here
        _logger.debug("Exception in is_option_supported: %s", e.what());
    }
    _logger.debug("option: %s is not supported", option.c_str());
    return false;
}

ov::SoPtr<IVCLCompiler> makeVCLCompiler(const std::string& libraryDir,
                                        const std::optional<IDevice::DeviceProperties>& deviceProperties) {
    auto api = VCLApi::getInstance(libraryDir);
    OPENVINO_ASSERT(api != nullptr, "VCL API table is nullptr");

    auto compiler = std::make_shared<VCLCompilerImpl>(api, deviceProperties);
    auto vclLib = compiler->getLinkedLibrary();
    OPENVINO_ASSERT(vclLib != nullptr, "VCL library is nullptr");

    // Pairing the compiler with the library keeps the .so alive for as long as the compiler is.
    return ov::SoPtr<IVCLCompiler>(compiler, vclLib);
}

}  // namespace intel_npu
