// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_impl.hpp"

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/vcl/vcl_allocator.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "model_serializer.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "vcl_error_utils.hpp"
#include "vcl_version_utils.hpp"
#include "weightless_utils.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

using vcl_version_utils::checkVclVersion;
using vcl_version_utils::getUsedVclVersion;
using vcl_version_utils::UsedVersion;

static std::optional<std::string> getVCLCompatibilityString(const VCLFunctionTable& functions,
                                                            vcl_executable_handle_t executable,
                                                            vcl_log_handle_t logHandle) {
    uint64_t compatibilityStringSize = 0;
    auto result = functions.vclExecutableGetCompatibilityString(executable, nullptr, &compatibilityStringSize);
    if (result == VCL_RESULT_ERROR_UNSUPPORTED_FEATURE) {
        return std::nullopt;
    }
    if (result != VCL_RESULT_SUCCESS || compatibilityStringSize == 0) {
        OPENVINO_THROW("Failed to get compatibility string size. vclExecutableGetCompatibilityString result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(functions, logHandle));
    }

    if (compatibilityStringSize > std::numeric_limits<size_t>::max()) {
        OPENVINO_THROW("Compatibility string size is too large to allocate a local buffer");
    }
    std::string compatibilityString(static_cast<size_t>(compatibilityStringSize), '\0');
    result =
        functions.vclExecutableGetCompatibilityString(executable, compatibilityString.data(), &compatibilityStringSize);
    if (result != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to get compatibility string. vclExecutableGetCompatibilityString result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(functions, logHandle));
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

VCLCompilerImpl::VCLCompilerImpl(std::shared_ptr<const VCLFunctionTable> functions,
                                 const std::optional<IDevice::DeviceProperties>& deviceProperties,
                                 ScopedOptionSupportCache optionSupportCache)
    : _functions(std::move(functions)),
      _logHandle(nullptr),
      _optionSupportCache(std::move(optionSupportCache)),
      _logger("VCLCompilerImpl", Logger::global().level()) {
    _logger.debug("VCLCompilerImpl constructor start");

    OPENVINO_ASSERT(_functions != nullptr, "VCLCompilerImpl requires a non-null VCLFunctionTable");
    OPENVINO_ASSERT(_functions->hasAllRequiredSymbols(),
                    "VCLCompilerImpl received a VCLFunctionTable with unresolved entry points. Was "
                    "it populated from a VCLLoader?");

    // Initialize the VCL API
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclGetVersion",
                          _functions->vclGetVersion(&_vclVersion, &_vclProfilingVersion),
                          nullptr);
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

    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclCompilerCreate",
                          _functions->vclCompilerCreate(&compilerDesc, &vclDeviceDesc, &_compilerHandle, &_logHandle),
                          nullptr);
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclCompilerGetProperties",
                          _functions->vclCompilerGetProperties(_compilerHandle, &_compilerProperties),
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
        vcl_result_t result = _functions->vclCompilerDestroy(_compilerHandle);
        _compilerHandle = nullptr;
        if (result != VCL_RESULT_SUCCESS) {
            _logger.warning("Failed to destroy VCL compiler: result 0x%x - %s",
                            result,
                            getLatestVCLLog(*_functions, _logHandle).c_str());
        }
    }

    if (_logHandle) {
        _logHandle = nullptr;  // Log handle is released automatically with the compiler
    }
    _logger.info("VCL Compiler destroyed successfully");
}

std::pair<ov::Tensor, std::optional<std::string>> VCLCompilerImpl::compile(
    const std::shared_ptr<const ov::Model>& model,
    const Config& config) const {
    return compile(model, config, false);
}

std::pair<ov::Tensor, std::optional<std::string>> VCLCompilerImpl::compile(
    const std::shared_ptr<const ov::Model>& model,
    const Config& config,
    const bool storeWeightlessCacheAttributeFlag) const {
    _logger.debug("compile start");

    /// Check the linked vcl version whether supported in plugin
    UsedVersion usedVersion =
        getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion.major, _vclVersion.minor);
    _logger.debug("The final vcl compiler version used is %d.%d", usedVersion.Major, usedVersion.Minor);
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

    const auto isOptionSupportedByCompiler = [this](const std::string& optionName,
                                                    const std::optional<std::string>& optionValue = std::nullopt) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionSupportedByCompiler,
                                                    false,
                                                    storeWeightlessCacheAttributeFlag);
    Config updatedConfig = config;
    if (is_option_supported(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update(ov::intel_npu::model_serializer_version.name(),
                             MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion));
    }

    std::string buildFlags;

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

    auto result = _functions->vclAllocatedExecutableCreate4(_compilerHandle,
                                                            exeDesc,
                                                            allocator.get(),
                                                            &blob,
                                                            &blobSize,
                                                            &executable);
    if (result != VCL_RESULT_SUCCESS) {
        // Check if allocations were performed before throwing exception
        auto tracked_allocations = allocator->m_info;
        for (const auto& [buffer, size] : tracked_allocations) {
            allocator->deallocate(allocator.get(), buffer);
        }
        if (executable != nullptr) {
            _functions->vclExecutableDestroy(executable);
        }
        OPENVINO_THROW("Compilation failed. vclAllocatedExecutableCreate4 result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(*_functions, _logHandle));
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
        compatibilityString = getVCLCompatibilityString(*_functions, executable, _logHandle);
    } catch (...) {
        _functions->vclExecutableDestroy(executable);
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

    result = _functions->vclExecutableDestroy(executable);
    if (result != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to destroy VCL executable. vclExecutableDestroy result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(*_functions, _logHandle));
    }

    return std::make_pair<ov::Tensor, std::optional<std::string>>(std::move(alignedBlob),
                                                                  std::move(compatibilityString));
}

std::pair<std::vector<ov::Tensor>, std::optional<std::string>> VCLCompilerImpl::compileWsOneShot(
    const std::shared_ptr<ov::Model>& model,
    const Config& config) const {
    _logger.debug("compileWsOneShot start");

    /// Check the linked vcl version whether supported in plugin
    UsedVersion usedVersion =
        getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion.major, _vclVersion.minor);
    _logger.debug("The final vcl compiler version used is %d.%d", usedVersion.Major, usedVersion.Minor);
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

    const auto isOptionSupportedByCompiler = [this](const std::string& optionName,
                                                    const std::optional<std::string>& optionValue = std::nullopt) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionSupportedByCompiler,
                                                    false,
                                                    true);
    Config updatedConfig = config;
    if (is_option_supported(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update(ov::intel_npu::model_serializer_version.name(),
                             MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion));
    }

    std::string buildFlags;

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

    auto result =
        _functions->vclAllocatedExecutableCreateWSOneShot2(_compilerHandle, exeDesc, allocator.get(), &executable);
    if (result != VCL_RESULT_SUCCESS) {
        if (executable != nullptr) {
            _functions->vclExecutableDestroy(executable);
        }
        OPENVINO_THROW("Compilation failed. vclAllocatedExecutableCreateWSOneShot2 result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(*_functions, _logHandle));
    }
    if (executable == nullptr) {
        OPENVINO_THROW("Failed to create VCL executable, executable handle is null");
    }

    if (allocator->m_info.size() == 0) {
        _functions->vclExecutableDestroy(executable);
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
        compatibilityString = getVCLCompatibilityString(*_functions, executable, _logHandle);
    } catch (...) {
        _functions->vclExecutableDestroy(executable);
        throw;
    }
    if (!compatibilityString.has_value()) {
        _logger.info("vclExecutableGetCompatibilityString is not supported for this executable (0x%x); "
                     "compatibility string will be absent",
                     uint32_t(VCL_RESULT_ERROR_UNSUPPORTED_FEATURE));
    } else {
        _logger.debug("Compatibility string from VCL: %s", compatibilityString->c_str());
    }

    result = _functions->vclExecutableDestroy(executable);
    if (result != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to destroy executable. vclExecutableDestroy result: 0x",
                       std::hex,
                       uint64_t(result),
                       " - ",
                       getLatestVCLLog(*_functions, _logHandle));
    }

    return std::make_pair(std::move(initMainTensors), std::move(compatibilityString));
}

std::pair<ov::Tensor, std::optional<std::string>> VCLCompilerImpl::compileWsIterative(
    const std::shared_ptr<ov::Model>& model,
    const Config& config,
    size_t callNumber) const {
    _logger.debug("compileWsIterative start");
    Config updatedConfig = config;
    updatedConfig.update(ov::intel_npu::ws_compile_call_number.name(), std::to_string(callNumber));
    // Return the compatibility descriptor together with the compiled blob.
    return compile(model, updatedConfig, true);
}

uint32_t VCLCompilerImpl::get_version() const {
    return ZE_MAKE_VERSION(_compilerProperties.version.major, _compilerProperties.version.minor);
}

ov::SupportedOpsMap VCLCompilerImpl::query(const std::shared_ptr<const ov::Model>& model, const Config& config) const {
    _logger.debug("query start");

    /// Check the linked vcl version whether supported in plugin
    UsedVersion usedVersion =
        getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion.major, _vclVersion.minor);
    _logger.debug("The final vcl compiler version used is %d.%d", usedVersion.Major, usedVersion.Minor);
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
    Config updatedConfig = config;
    const auto isOptionSupportedByCompiler = [this](const std::string& optionName,
                                                    const std::optional<std::string>& optionValue = std::nullopt) {
        return is_option_supported(optionName, optionValue);
    };
    auto serializedIR = compiler_utils::serializeIR(model,
                                                    compilerVersion,
                                                    maxOpsetVersion,
                                                    config.get<MODEL_SERIALIZER_VERSION>(),
                                                    isOptionSupportedByCompiler);
    if (is_option_supported(ov::intel_npu::model_serializer_version.name())) {
        updatedConfig.update(ov::intel_npu::model_serializer_version.name(),
                             MODEL_SERIALIZER_VERSION::toString(serializedIR.serializerVersion));
    }

    std::string buildFlags;
    buildFlags += compiler_utils::serializeConfig(updatedConfig, compilerVersion, isOptionSupportedByCompiler);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    vcl_query_handle_t queryHandle;
    vcl_query_desc_t queryDesc = {serializedIR.buffer.get(), serializedIR.size, buildFlags.c_str(), buildFlags.size()};
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclQueryNetworkCreate",
                          _functions->vclQueryNetworkCreate(_compilerHandle, queryDesc, &queryHandle),
                          _logHandle);

    uint64_t size = 0;
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclQueryNetwork",
                          _functions->vclQueryNetwork(queryHandle, nullptr, &size),
                          _logHandle);

    std::vector<char> supportedLayers(size);
    THROW_ON_FAIL_FOR_VCL(
        *_functions,
        "vclQueryNetwork",
        _functions->vclQueryNetwork(queryHandle, reinterpret_cast<uint8_t*>(supportedLayers.data()), &size),
        _logHandle);

    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclQueryNetworkDestroy",
                          _functions->vclQueryNetworkDestroy(queryHandle),
                          _logHandle);

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
    uint64_t str_size = 0;
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclGetCompilerSupportedOptions",
                          _functions->vclGetCompilerSupportedOptions(_compilerHandle, nullptr, &str_size),
                          _logHandle);

    if (str_size == 0) {
        _logger.debug("Option list size 0!");
        return {};
    }

    _logger.debug("obtain list");
    std::vector<char> options(str_size);
    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclGetCompilerSupportedOptions",
                          _functions->vclGetCompilerSupportedOptions(_compilerHandle, options.data(), &str_size),
                          _logHandle);

    _logger.debug("Option list size %" PRIu64 ", got option list", str_size);

    size_t optionsSize = options.size();
    while (optionsSize > 0 && options[optionsSize - 1] == '\0') {
        --optionsSize;
    }
    if (optionsSize == 0) {
        return {};
    }

    std::string compilerOptionsStr(options.data(), optionsSize);
    _logger.debug("VCLCompilerImpl return supported_options: %s", compilerOptionsStr.c_str());
    // vectorize string
    std::istringstream suppstream(compilerOptionsStr);
    std::vector<std::string> compilerOpts;
    std::string option;
    while (suppstream >> option) {
        compilerOpts.push_back(option);
    }

    _optionSupportCache.setSupportedOptions(compilerOpts);

    return compilerOpts;
}

bool VCLCompilerImpl::is_option_supported(const std::string& option, const std::optional<std::string>& optValue) const {
    // The cache is keyed by option name alone, so it can only answer queries that do not carry a value.
    const bool useCache = !optValue.has_value();
    if (useCache) {
        const auto cachedSupport = _optionSupportCache.isOptionSupported(option);
        if (cachedSupport.has_value()) {
            return cachedSupport.value();
        }
    }

    bool supported = false;
    try {
        const char* optname_ch = option.c_str();
        const char* optvalue_ch = optValue.has_value() ? optValue.value().c_str() : nullptr;
        THROW_ON_FAIL_FOR_VCL(*_functions,
                              "vclGetCompilerIsOptionSupported",
                              _functions->vclGetCompilerIsOptionSupported(_compilerHandle, optname_ch, optvalue_ch),
                              _logHandle);
        supported = true;
    } catch (const std::exception& e) {
        // The API is only supported in new version, just add log here
        _logger.debug("Exception in is_option_supported: %s", e.what());
    }

    if (useCache) {
        _optionSupportCache.addSupportedOption(option, supported);
    }

    return supported;
}

namespace {
// The cache key under which the compiler-in-plugin's option-support answers are stored.
constexpr OptionSupportCache::CacheKey pluginOptionSupportKey =
    static_cast<OptionSupportCache::CacheKey>(ov::intel_npu::CompilerType::PLUGIN);
}  // namespace

ov::SoPtr<IVCLCompiler> makeVCLCompiler(const std::optional<IDevice::DeviceProperties>& deviceProperties,
                                        const std::shared_ptr<OptionSupportCache>& optionSupportCache) {
    auto vclLoader = VCLLoader::getInstance();
    OPENVINO_ASSERT(vclLoader != nullptr, "VCL loader is nullptr");

    auto compiler =
        std::make_shared<VCLCompilerImpl>(vclLoader->sharedFunctions(),
                                          deviceProperties,
                                          ScopedOptionSupportCache{optionSupportCache, pluginOptionSupportKey});

    // Pairing the compiler with the library keeps the .so alive for as long as the compiler is. The
    // compiler itself never learns that a library is involved.
    auto vclLib = vclLoader->getLibrary();
    OPENVINO_ASSERT(vclLib != nullptr, "VCL library is nullptr");

    return ov::SoPtr<IVCLCompiler>(compiler, vclLib);
}

}  // namespace intel_npu
