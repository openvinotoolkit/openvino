// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_impl.hpp"

#include <limits>
#include <mutex>

#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/profiling.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "model_serializer.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "weightless_utils.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace {

struct UsedVersion {
    int Major;
    int Minor;
    UsedVersion(int major, int minor) : Major(major), Minor(minor) {}
};

UsedVersion getUsedVclVersion(uint16_t pluginMajor, uint16_t pluginMinor, const vcl_version_info_t& loadedVersion) {
    uint16_t usedMajor = pluginMajor, usedMinor = pluginMinor;
    if (pluginMajor == loadedVersion.major) {
        usedMinor = std::min(pluginMinor, loadedVersion.minor);
    } else if (pluginMajor > loadedVersion.major) {
        usedMajor = loadedVersion.major;
        usedMinor = loadedVersion.minor;
    }
    return {usedMajor, usedMinor};
}

struct vcl_allocator : vcl_allocator2_t {
    vcl_allocator() : vcl_allocator2_t{allocate, deallocate} {}

    static uint8_t* allocate(vcl_allocator2_t* allocator, size_t size) {
        vcl_allocator* vclAllocator = static_cast<vcl_allocator*>(allocator);
        vclAllocator->m_size = intel_npu::utils::align_size_to_standard_page_size(size);
        auto allocatedPtr = reinterpret_cast<uint8_t*>(
            vclAllocator->m_allocator.allocate(vclAllocator->m_size, intel_npu::utils::STANDARD_PAGE_SIZE));
        if (allocatedPtr == nullptr) {
            OPENVINO_THROW("Failed to allocate aligned memory for allocator");
        }
        memset(allocatedPtr + size, 0, vclAllocator->m_size - size);
        vclAllocator->m_allocated = allocatedPtr;
        return allocatedPtr;
    }

    static void deallocate(vcl_allocator2_t* allocator, uint8_t* ptr) {
        if (ptr == nullptr) {
            OPENVINO_THROW("Pointer is nullptr in deallocate!");
        }
        vcl_allocator* vclAllocator = static_cast<vcl_allocator*>(allocator);
        vclAllocator->m_allocator.deallocate(ptr, vclAllocator->m_size, intel_npu::utils::STANDARD_PAGE_SIZE);
    }
    ov::Allocator m_allocator;
    uint8_t* m_allocated = nullptr;
    size_t m_size = 0;
};

struct vcl_allocator_2 : vcl_allocator2_t {
    vcl_allocator_2() : vcl_allocator2_t{allocate, deallocate} {}

    static uint8_t* allocate(vcl_allocator2_t* allocator, size_t size) {
        vcl_allocator_2* vclAllocator = static_cast<vcl_allocator_2*>(allocator);
        size_t alignedSize = intel_npu::utils::align_size_to_standard_page_size(size);
        auto allocatedPtr = reinterpret_cast<uint8_t*>(
            vclAllocator->m_allocator.allocate(alignedSize, intel_npu::utils::STANDARD_PAGE_SIZE));
        if (allocatedPtr == nullptr) {
            OPENVINO_THROW("Failed to allocate aligned memory for allocator");
        }
        memset(allocatedPtr + size, 0, alignedSize - size);
        vclAllocator->m_info.emplace_back(std::make_pair(allocatedPtr, alignedSize));
        return allocatedPtr;
    }

    static void deallocate(vcl_allocator2_t* allocator, uint8_t* ptr) {
        if (ptr == nullptr) {
            OPENVINO_THROW("Pointer is nullptr in deallocate!");
        }
        vcl_allocator_2* vclAllocator = static_cast<vcl_allocator_2*>(allocator);
        // 1 is the placeholder value, as size is not needed in deallocate
        vclAllocator->m_allocator.deallocate(ptr, 1, intel_npu::utils::STANDARD_PAGE_SIZE);
    }
    ov::Allocator m_allocator;
    std::vector<std::pair<uint8_t*, size_t>> m_info;
};

ov::Tensor make_tensor_from_aligned_addr(uint8_t* allocated, size_t size) {
    ov::Allocator allocator;
    auto tensor = ov::Tensor(ov::element::u8, ov::Shape{size}, allocated);
    auto impl = ov::get_tensor_impl(std::move(tensor));
    std::shared_ptr<void> ptr(allocated, [allocator, size](uint8_t* p) mutable {
        if (p == nullptr) {
            OPENVINO_THROW("Pointer is nullptr in memory deallocation of make_tensor_from_aligned_addr!");
        }
        allocator.deallocate(p, size, intel_npu::utils::STANDARD_PAGE_SIZE);
    });
    impl._so = ptr;
    return ov::make_tensor(impl);
}

}  // namespace

namespace intel_npu {

static inline std::string getLatestVCLLog(vcl_log_handle_t logHandle) {
    Logger _logger("VCLAPI", Logger::global().level());
    _logger.debug("getLatestVCLLog start");

    vcl_version_info_t compilerVersion;
    vcl_version_info_t profilingVersion;
    vcl_result_t ret = vclGetVersion(&compilerVersion, &profilingVersion);

    if (ret != VCL_RESULT_SUCCESS || compilerVersion.major < 3) {
        _logger.warning("Failed to get VCL version: 0x%x", ret);
        return "Can not get VCL log, VCL version is too old!";
    }

    // Get log size
    size_t size = 0;
    // Null graph handle to get error log
    ret = vclLogHandleGetString(logHandle, &size, nullptr);
    if (VCL_RESULT_SUCCESS != ret) {
        return "Failed to get size of latest VCL log";
    }

    if (size <= 0) {
        return "No error stored in VCL when error detected";
    }

    // Get log content
    std::string logContent{};
    logContent.resize(size);
    ret = vclLogHandleGetString(logHandle, &size, const_cast<char*>(logContent.data()));
    if (VCL_RESULT_SUCCESS != ret) {
        return "Size of latest error log > 0, failed to get content";
    }
    _logger.debug("getLatestBuildError end");
    return logContent;
}

#define THROW_ON_FAIL_FOR_VCL(step, ret, logHandle)     \
    {                                                   \
        vcl_result_t result = ret;                      \
        if (result != VCL_RESULT_SUCCESS) {             \
            OPENVINO_THROW("Failed to call VCL API : ", \
                           step,                        \
                           " result: 0x",               \
                           std::hex,                    \
                           result,                      \
                           " - ",                       \
                           getLatestVCLLog(logHandle)); \
        }                                               \
    }

const std::shared_ptr<VCLCompilerImpl> VCLCompilerImpl::getInstance() {
    static std::mutex mutex;
    static std::weak_ptr<VCLCompilerImpl> weak_compiler;

    std::lock_guard<std::mutex> lock(mutex);
    auto compiler = weak_compiler.lock();
    if (!compiler) {
        compiler = std::make_shared<VCLCompilerImpl>();
        weak_compiler = compiler;
    }
    return compiler;
}

VCLCompilerImpl::VCLCompilerImpl() : _logHandle(nullptr), _logger("VCLCompilerImpl", Logger::global().level()) {
    _logger.debug("VCLCompilerImpl constructor start");

    // Load VCL library
    (void)VCLApi::getInstance();

    // Initialize the VCL API
    THROW_ON_FAIL_FOR_VCL("vclGetVersion", vclGetVersion(&_vclVersion, &_vclProfilingVersion), nullptr);
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

    // This information cannot be determined during the initialization phase; set device desc default value, the related
    // info will be processed in compile phase if passed by user.
    _logger.info("Device description is not provided, using default values");
    uint32_t defaultTileCount = std::numeric_limits<uint32_t>::max();
    if (_vclVersion.major == 7 && _vclVersion.minor < 6) {
        // For vcl <= 7.5, need to use smaller value to pass check
        defaultTileCount = std::numeric_limits<uint16_t>::max();
    }
    vcl_device_desc_t device_desc = {sizeof(vcl_device_desc_t),
                                     0x00,
                                     std::numeric_limits<uint16_t>::max(),
                                     defaultTileCount};
    THROW_ON_FAIL_FOR_VCL("vclCompilerCreate",
                          vclCompilerCreate(&compilerDesc, &device_desc, &_compilerHandle, &_logHandle),
                          nullptr);
    THROW_ON_FAIL_FOR_VCL("vclCompilerGetProperties",
                          vclCompilerGetProperties(_compilerHandle, &_compilerProperties),
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
        vcl_result_t result = vclCompilerDestroy(_compilerHandle);
        _compilerHandle = nullptr;
        if (result != VCL_RESULT_SUCCESS) {
            _logger.warning("Failed to destroy VCL compiler: result 0x%x - %s",
                            result,
                            getLatestVCLLog(_logHandle).c_str());
        }
    }

    if (_logHandle) {
        _logHandle = nullptr;  // Log handle is released automatically with the compiler
    }
    _logger.info("VCL Compiler destroyed successfully");
}

std::shared_ptr<void> VCLCompilerImpl::getLinkedLibrary() const {
    return VCLApi::getInstance()->getLibrary();
}

ov::Tensor VCLCompilerImpl::compile(const std::shared_ptr<const ov::Model>& model, const FilteredConfig& config) const {
    return compile(model, config, false);
}

ov::Tensor VCLCompilerImpl::compile(const std::shared_ptr<const ov::Model>& model,
                                    const FilteredConfig& config,
                                    const bool storeWeightlessCacheAttributeFlag) const {
    _logger.debug("compile start");

    /// Check the linked vcl version whether supported in plugin
    UsedVersion usedVersion = getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion);
    _logger.debug("the finally used compiler vcl version is %d.%d", usedVersion.Major, usedVersion.Minor);

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

    if (usedVersion.Major >= 7 && usedVersion.Minor >= 4) {
        // support the lastest vcl api
        // For VCL 7.4 and later, we can use vclAllocatedExecutableCreate2
        _logger.debug("Using vclAllocatedExecutableCreate2 for 7.4 <= VCL");
        vcl_allocator allocator;
        uint8_t* blob = nullptr;
        size_t size = 0;

        auto result = vclAllocatedExecutableCreate2(_compilerHandle, exeDesc, &allocator, &blob, &size);
        if (result != VCL_RESULT_SUCCESS) {
            OPENVINO_THROW("Compilation failed. vclAllocatedExecutableCreate2 result: 0x",
                           std::hex,
                           uint64_t(result),
                           " - ",
                           getLatestVCLLog(_logHandle));
        }

        if (size == 0 || blob == nullptr) {
            OPENVINO_THROW("Failed to create VCL executable, size is zero or blob is null");
        }
        // The allocated size from VCL will be equal or smaller than the allocated size in allocator
        _logger.debug("Blob size from VCL: %zu ptr %p", size, static_cast<void*>(blob));
        _logger.debug("Allocated vector size: %zu ptr: %p",
                      allocator.m_size,
                      static_cast<void*>(allocator.m_allocated));

        _logger.debug("compile end, blob size:%d", allocator.m_size);
        return make_tensor_from_aligned_addr(allocator.m_allocated, allocator.m_size);
    } else {
        OPENVINO_THROW("Not supported VCL version: %d.%d, please use VCL 6.1 or later",
                       _vclVersion.major,
                       _vclVersion.minor);
    }
}

std::vector<ov::Tensor> VCLCompilerImpl::compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                                          const FilteredConfig& config) const {
    _logger.debug("compileWsOneShot start");

    /// Check the linked vcl version whether supported in plugin
    UsedVersion usedVersion = getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion);
    _logger.debug("the finally used compiler vcl version is %d.%d", usedVersion.Major, usedVersion.Minor);

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

    _logger.debug("Using vclAllocatedExecutableCreateWSOneShot");
    vcl_allocator_2 allocator;

    THROW_ON_FAIL_FOR_VCL("vclAllocatedExecutableCreateWSOneShot",
                          vclAllocatedExecutableCreateWSOneShot(_compilerHandle, exeDesc, &allocator),
                          _logHandle);

    if (allocator.m_info.size() == 0) {
        OPENVINO_THROW("Failed to create VCL executable, blobCount is zero");
    }

    std::vector<ov::Tensor> initMainTensors;
    for (auto& blob : allocator.m_info) {
        initMainTensors.emplace_back(make_tensor_from_aligned_addr(blob.first, blob.second));
    }
    return initMainTensors;
}

ov::Tensor VCLCompilerImpl::compileWsIterative(const std::shared_ptr<ov::Model>& model,
                                               const FilteredConfig& config,
                                               size_t callNumber) const {
    _logger.debug("compileWsIterative start");
    FilteredConfig updatedConfig = config;
    updatedConfig.update({{ov::intel_npu::ws_compile_call_number.name(), std::to_string(callNumber)}});
    return compile(model, updatedConfig, true);
}

std::vector<ov::ProfilingInfo> VCLCompilerImpl::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                         const std::vector<uint8_t>& network) const {
    _logger.debug("process_profiling_output start");

    vcl_profiling_handle_t profilingHandle;
    vcl_profiling_input_t profilingInput = {network.data(), network.size(), profData.data(), profData.size()};
    vcl_log_handle_t logHandle;
    THROW_ON_FAIL_FOR_VCL("vclProfilingCreate",
                          vclProfilingCreate(&profilingInput, &profilingHandle, &logHandle),
                          nullptr);

    vcl_profiling_properties_t profProperties;
    THROW_ON_FAIL_FOR_VCL("vclProfilingGetProperties",
                          vclProfilingGetProperties(profilingHandle, &profProperties),
                          logHandle);

    _logger.info("VCL Profiling Properties: Version: %d.%d",
                 profProperties.version.major,
                 profProperties.version.minor);

    // We only use layer level info
    vcl_profiling_request_type_t request = VCL_PROFILING_LAYER_LEVEL;

    vcl_profiling_output_t profOutput;
    profOutput.data = NULL;
    THROW_ON_FAIL_FOR_VCL("vclGetDecodedProfilingBuffer",
                          vclGetDecodedProfilingBuffer(profilingHandle, request, &profOutput),
                          logHandle);
    if (profOutput.data == NULL) {
        OPENVINO_THROW("Failed to get VCL profiling output");
    }

    std::vector<ze_profiling_layer_info> layerInfo(profOutput.size / sizeof(ze_profiling_layer_info));
    if (profOutput.size > 0) {
        _logger.debug("VCL profiling output size: %d", profOutput.size);
        std::memcpy(layerInfo.data(), profOutput.data, profOutput.size);
    }

    THROW_ON_FAIL_FOR_VCL("vclProfilingDestroy", vclProfilingDestroy(profilingHandle), logHandle);

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
    UsedVersion usedVersion = getUsedVclVersion(VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR, _vclVersion);
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
                          vclQueryNetworkCreate(_compilerHandle, queryDesc, &queryHandle),
                          _logHandle);

    uint64_t size = 0;
    THROW_ON_FAIL_FOR_VCL("vclQueryNetwork", vclQueryNetwork(queryHandle, nullptr, &size), _logHandle);

    std::vector<char> supportedLayers(size);
    THROW_ON_FAIL_FOR_VCL("vclQueryNetwork",
                          vclQueryNetwork(queryHandle, reinterpret_cast<uint8_t*>(supportedLayers.data()), &size),
                          _logHandle);

    THROW_ON_FAIL_FOR_VCL("vclQueryNetworkDestroy", vclQueryNetworkDestroy(queryHandle), _logHandle);

    const std::string deviceName = "NPU";
    ov::SupportedOpsMap result;
    const auto parsedSupportedLayers = parseQueryResult(supportedLayers);
    for (auto&& layerName : parsedSupportedLayers) {
        result.emplace(layerName, deviceName);
    }
    _logger.info("For given model, there are %d supported layers", parsedSupportedLayers.size());

    return result;
}

bool VCLCompilerImpl::get_supported_options(std::vector<char>& options) const {
    _logger.debug("get_supported_options start");
    size_t str_size = 0;
    try {
        THROW_ON_FAIL_FOR_VCL("vclGetCompilerSupportedOptions",
                              vclGetCompilerSupportedOptions(_compilerHandle, nullptr, &str_size),
                              _logHandle);

        if (str_size == 0) {
            _logger.debug("Option list size 0!");
            return true;
        }

        _logger.debug("obtain list");
        options.resize(str_size);
        THROW_ON_FAIL_FOR_VCL("vclGetCompilerSupportedOptions",
                              vclGetCompilerSupportedOptions(_compilerHandle, options.data(), &str_size),
                              _logHandle);

        _logger.debug("Option list size %d, got option list", str_size);

        return true;
    } catch (const std::exception& e) {
        // The API is only supported in new version, just add log here
        _logger.debug("Exception in get_supported_options: %s", e.what());
    }

    return false;
}

bool VCLCompilerImpl::is_option_supported(std::string option, std::optional<std::string> optValue) const {
    try {
        const char* optname_ch = option.c_str();
        const char* optvalue_ch = optValue.has_value() ? optValue.value().c_str() : nullptr;
        _logger.debug("is_option_supported start for option: %s, value: %s",
                      optname_ch,
                      optValue ? optvalue_ch : "null");
        THROW_ON_FAIL_FOR_VCL("vclGetCompilerIsOptionSupported",
                              vclGetCompilerIsOptionSupported(_compilerHandle, optname_ch, optvalue_ch),
                              _logHandle);
        return true;
    } catch (const std::exception& e) {
        // The API is only supported in new version, just add log here
        _logger.debug("Exception in is_option_supported: %s", e.what());
    }
    _logger.debug("option: %s is not supported", option.c_str());
    return false;
}

std::vector<uint8_t> VCLCompilerImpl::get_compiled_model_compatibility_descriptor() const {
    // TODO use the new call
    vcl_allocator allocator;
    uint8_t* blob = nullptr;
    size_t size = 0;

    auto result = vclAllocatedExecutableCreate3(_compilerHandle,
                                                exeDesc,
                                                &allocator,
                                                &blob,
                                                &size,
                                                &compatibilityDescriptor,
                                                &descriptorSize);
}

bool VCLCompilerImpl::validate_compatibility_descriptor(const std::string& compatibilityDescriptor) const {
    // TODO use the new call
}

}  // namespace intel_npu
