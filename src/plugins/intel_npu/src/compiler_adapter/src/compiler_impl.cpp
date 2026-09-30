// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_impl.hpp"

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
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

static bool hasCompilationTargetSymbols(const VCLFunctionTable& functions) {
    return functions.vclCompilationTargetsCreate != nullptr && functions.vclCompilationTargetsGetCount != nullptr &&
           functions.vclCompilationTargetsGetTarget != nullptr &&
           functions.vclCompilationTargetGetPlatform != nullptr &&
           functions.vclCompilationTargetGetConfigBundleCount != nullptr &&
           functions.vclCompilationTargetGetConfigBundle != nullptr &&
           functions.vclCompilationTargetsDestroy != nullptr;
}

namespace {

struct ResolvedCompilationTarget {
    std::string platform;
    std::vector<std::string> configBundles;
};

}  // namespace

/**
 * @brief Resolves \p config through the VCL compilation target entry points and copies every target out of the
 * handle, which owns the strings and is destroyed before returning.
 */
static std::vector<ResolvedCompilationTarget> resolveVCLCompilationTargets(const VCLFunctionTable& functions,
                                                                           const std::string& config,
                                                                           const Logger& logger) {
    vcl_compilation_targets_handle_t createdHandle = nullptr;
    // No log handle on purpose: a failed create returns none, so a requested one would swallow the
    // compiler's reason. Without it the compiler prints the reason itself. The getters log nothing.
    THROW_ON_FAIL_FOR_VCL(functions,
                          "vclCompilationTargetsCreate",
                          functions.vclCompilationTargetsCreate(config.empty() ? nullptr : config.c_str(),
                                                                config.size(),
                                                                &createdHandle,
                                                                nullptr),
                          nullptr);

    const auto destroyTargets = [&functions, &logger](vcl_compilation_targets_handle_t handle) {
        const vcl_result_t destroyResult = functions.vclCompilationTargetsDestroy(handle);
        if (destroyResult != VCL_RESULT_SUCCESS) {
            logger.warning("Failed to destroy VCL compilation targets handle: result 0x%x", destroyResult);
        }
    };
    std::unique_ptr<std::remove_pointer_t<vcl_compilation_targets_handle_t>, decltype(destroyTargets)> targetsHandle(
        createdHandle,
        destroyTargets);

    uint64_t targetCount = 0;
    THROW_ON_FAIL_FOR_VCL(functions,
                          "vclCompilationTargetsGetCount",
                          functions.vclCompilationTargetsGetCount(targetsHandle.get(), &targetCount),
                          nullptr);

    std::vector<ResolvedCompilationTarget> targets;
    targets.reserve(static_cast<size_t>(targetCount));
    for (uint64_t targetIndex = 0; targetIndex < targetCount; ++targetIndex) {
        vcl_compilation_target_handle_t targetHandle = nullptr;
        THROW_ON_FAIL_FOR_VCL(functions,
                              "vclCompilationTargetsGetTarget",
                              functions.vclCompilationTargetsGetTarget(targetsHandle.get(), targetIndex, &targetHandle),
                              nullptr);

        ResolvedCompilationTarget& target = targets.emplace_back();

        const char* platform = nullptr;
        uint64_t platformSize = 0;
        THROW_ON_FAIL_FOR_VCL(functions,
                              "vclCompilationTargetGetPlatform",
                              functions.vclCompilationTargetGetPlatform(targetHandle, &platform, &platformSize),
                              nullptr);
        OPENVINO_ASSERT(platform != nullptr, "vclCompilationTargetGetPlatform returned a null platform");
        target.platform.assign(platform, static_cast<size_t>(platformSize));

        uint64_t bundleCount = 0;
        THROW_ON_FAIL_FOR_VCL(functions,
                              "vclCompilationTargetGetConfigBundleCount",
                              functions.vclCompilationTargetGetConfigBundleCount(targetHandle, &bundleCount),
                              nullptr);
        target.configBundles.reserve(static_cast<size_t>(bundleCount));
        for (uint64_t bundleIndex = 0; bundleIndex < bundleCount; ++bundleIndex) {
            const char* bundle = nullptr;
            uint64_t bundleSize = 0;
            THROW_ON_FAIL_FOR_VCL(
                functions,
                "vclCompilationTargetGetConfigBundle",
                functions.vclCompilationTargetGetConfigBundle(targetHandle, bundleIndex, &bundle, &bundleSize),
                nullptr);
            OPENVINO_ASSERT(bundle != nullptr, "vclCompilationTargetGetConfigBundle returned a null bundle");
            target.configBundles.emplace_back(bundle, static_cast<size_t>(bundleSize));
        }
    }

    return targets;
}

VCLCompilerImpl::VCLCompilerImpl(const std::shared_ptr<const VCLFunctionTable>& functions,
                                 const std::optional<IDevice::DeviceProperties>& deviceProperties,
                                 ScopedOptionSupportCache optionSupportCache)
    : _functions(functions),
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

    std::optional<vcl_device_desc_t> vclDeviceDesc;
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
        vclDeviceDesc = vcl_device_desc_t{sizeof(vcl_device_desc_t),
                                          deviceProperties->deviceId,
                                          revision,
                                          deviceProperties->numSlices};
    } else {
        _logger.info("Device description is not provided, passing a null device descriptor");
    }

    THROW_ON_FAIL_FOR_VCL(*_functions,
                          "vclCompilerCreate",
                          _functions->vclCompilerCreate(&compilerDesc,
                                                        vclDeviceDesc.has_value() ? &vclDeviceDesc.value() : nullptr,
                                                        &_compilerHandle,
                                                        &_logHandle),
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

std::vector<std::string> VCLCompilerImpl::resolve_compilation_target_bundles(const Config& config) const {
    _logger.debug("resolve_compilation_target_bundles start");

    // Optional/weak entry points: an older VCL library cannot resolve SKU variants, so no bundle is
    // available and the caller rejects the compilation target.
    if (!hasCompilationTargetSymbols(*_functions)) {
        return {};
    }

    // The full config is sent as-is to the resolution query. The compiler ignores the options it knows but does
    // not need for resolution; an option it does not know at all makes the query fail with INVALID_ARGUMENT.
    const std::string resolutionConfig = config.toStringForCompiler([](const std::string&) {
        return true;
    });

    // Alternative under discussion: filter on the plugin side, the same way a compile does. The query rejects any
    // option the compiler does not know, so with the accept-everything filter above an unsupported RunTime/Both
    // option, or NPU_COMPILE_LOG_LEVEL (the compiler knows only LOG_LEVEL), fails the resolution even though the
    // compile itself would accept the config. This variant skips/reports unsupported options exactly as a compile
    // does and maps NPU_COMPILE_LOG_LEVEL onto LOG_LEVEL as compiler_utils::serializeConfig does
    // (compiler_utils::serializeConfig itself cannot be reused: it prefixes its output with "--config ").
    //
    // std::optional<Config> configWithCompileLogLevel;
    // if (config.has<COMPILE_LOG_LEVEL>()) {
    //     configWithCompileLogLevel = config;
    //     configWithCompileLogLevel->update(ov::log::level.name(), config.get<COMPILE_LOG_LEVEL>());
    //     configWithCompileLogLevel->remove(ov::intel_npu::compile_log_level.name());
    // }
    // const Config& filteredConfig = configWithCompileLogLevel.has_value() ? *configWithCompileLogLevel : config;
    // const std::string resolutionConfig = filteredConfig.toStringForCompiler([this](const std::string& optionName) {
    //     return is_option_supported(optionName);
    // });

    _logger.info("Resolving compilation target bundles for config '%s'", resolutionConfig.c_str());

    // NPU_PLATFORM is always present in resolutionConfig here (resolveCompilationTarget already
    // injected it), so the query narrows to that one platform - at most one target comes back.
    const auto targets = resolveVCLCompilationTargets(*_functions, resolutionConfig, _logger);
    OPENVINO_ASSERT(targets.size() <= 1,
                    "The compiler resolved ",
                    targets.size(),
                    " compilation targets for config '",
                    resolutionConfig,
                    "', which names one platform; expected at most one");
    if (!targets.empty()) {
        OPENVINO_ASSERT(ov::intel_npu::Platform::standardize(targets.front().platform) ==
                            ov::intel_npu::Platform::standardize(config.get<PLATFORM>()),
                        "The compiler resolved a compilation target for platform '",
                        targets.front().platform,
                        "' while platform '",
                        config.get<PLATFORM>(),
                        "' was requested");
    }
    std::vector<std::string> bundles = targets.empty() ? std::vector<std::string>{} : targets.front().configBundles;

    // A successful query that resolves to no target means nothing can be compiled for this request: the
    // platform is unknown to the compiler, or does not support an option the config asks for.
    OPENVINO_ASSERT(!bundles.empty(),
                    "Cannot compile for platform '",
                    config.get<PLATFORM>(),
                    "': the compiler resolved no compilation target for config '",
                    resolutionConfig,
                    "'. The platform is unknown to this compiler, or does not support an option of the config.");

    std::ostringstream bundlesLog;
    bundlesLog << "Resolved " << bundles.size() << " compilation target bundle(s): ";
    for (size_t i = 0; i < bundles.size(); ++i) {
        if (i > 0) {
            bundlesLog << "; ";
        }
        bundlesLog << '[' << i << "] '" << bundles[i] << '\'';
    }
    _logger.info("%s", bundlesLog.str().c_str());

    return bundles;
}

std::vector<std::string> VCLCompilerImpl::get_offline_compilation_targets() const {
    _logger.debug("get_offline_compilation_targets start");

    if (!hasCompilationTargetSymbols(*_functions)) {
        _logger.info("The VCL library does not export the compilation target entry points");
        return {};
    }

    // No config leaves NPU_PLATFORM out, so every platform the compiler can compile for resolves to a target.
    std::vector<std::string> result;
    for (auto& target : resolveVCLCompilationTargets(*_functions, std::string{}, _logger)) {
        result.push_back(std::move(target.platform));
    }

    return result;
}

}  // namespace intel_npu
