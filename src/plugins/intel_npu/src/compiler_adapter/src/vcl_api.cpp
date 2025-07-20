// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_api.hpp"

#include "intel_npu/profiling.hpp"
#include "ir_serializer.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

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

VCLApi::VCLApi() : _logger("VCLApi", ov::log::Level::DEBUG) {
    const std::string baseName = "npu_vcl_compiler";
    try {
        auto libpath = ov::util::make_plugin_library_name({}, baseName);
        _logger.debug("Try to load npu_vcl_compiler");

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        this->lib = ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
        this->lib = ov::util::load_shared_object(libpath.c_str());
#endif
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to load npu_vcl_compiler");
        OPENVINO_THROW(error.what());
    }

    try {
#define vcl_symbol_statement(vcl_symbol) \
    this->vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol));
        vcl_symbols_list();
#undef vcl_symbol_statement
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to get formal symbols from npu_vcl_compiler");
        OPENVINO_THROW(error.what());
    }

#define vcl_symbol_statement(vcl_symbol)                                                                      \
    try {                                                                                                     \
        this->vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol)); \
    } catch (const std::runtime_error&) {                                                                     \
        _logger.debug("Failed to get %s from npu_vcl_compiler", #vcl_symbol);                                 \
        this->vcl_symbol = nullptr;                                                                           \
    }
    vcl_weak_symbols_list();
#undef vcl_symbol_statement

#define vcl_symbol_statement(vcl_symbol) vcl_symbol = this->vcl_symbol;
    vcl_symbols_list();
    vcl_weak_symbols_list();
#undef vcl_symbol_statement
}

const std::shared_ptr<VCLApi>& VCLApi::getInstance() {
    static std::shared_ptr<VCLApi> instance = std::make_shared<VCLApi>();
    return instance;
}

VCLCompilerImpl::VCLCompilerImpl() : _logHandle(nullptr), _logger("VCLCompilerImpl", ov::log::Level::DEBUG) {
    _logger.debug("VCLCompilerImpl constructor start");
    // Initialize the VCL API
    THROW_ON_FAIL_FOR_VCL("vclGetVersion", vclGetVersion(&_vclVersion, &_vclProfilingVersion), nullptr);

    _logger.info("Plugin VCL API Version: %d.%d", VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR);
    _logger.info("Plugin VCL Profiling API Version: %d.%d", VCL_PROFILING_VERSION_MAJOR, VCL_PROFILING_VERSION_MINOR);
    _logger.info("Lib VCL Compiler Version: %d.%d", _vclVersion.major, _vclVersion.minor);
    _logger.info("Lib VCL Profiling Version: %d.%d", _vclProfilingVersion.major, _vclProfilingVersion.minor);
    _logger.info("Use Lib VCL version to create compiler");

    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version = _vclVersion;
    compilerDesc.debugLevel = static_cast<__vcl_log_level_t>(static_cast<int>(Logger::global().level()) - 1);
    vcl_device_desc_t device_desc;
    device_desc.size = sizeof(vcl_device_desc_t);
    device_desc.deviceID = 0x643E;  // Value from intel_npu/src/backend/src/zero_device.cpp
    device_desc.revision = -1;      // -1 to skip the config
    device_desc.tileCount = 5;      // 1 as init value

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
        THROW_ON_FAIL_FOR_VCL("vclCompilerDestroy", vclCompilerDestroy(_compilerHandle), _logHandle);
    }
    if (_logHandle) {
        _logHandle = nullptr;  // Log handle is released automatically with the compiler
    }
    _logger.info("VCL Compiler destroyed successfully");
}

struct vcl_allocator_vector : vcl_allocator2_t {
    vcl_allocator_vector() : vcl_allocator2_t{vector_allocate, vector_deallocate} {}

    static uint8_t* vector_allocate(vcl_allocator2_t* allocator, size_t size) {
        vcl_allocator_vector* vecAllocator = static_cast<vcl_allocator_vector*>(allocator);
        vecAllocator->m_vec.resize(size);
        return vecAllocator->m_vec.data();
    }

    static void vector_deallocate(vcl_allocator2_t* allocator, uint8_t* ptr) {
        vcl_allocator_vector* vecAllocator = static_cast<vcl_allocator_vector*>(allocator);
        vecAllocator->m_vec.clear();
        vecAllocator->m_vec.shrink_to_fit();
    }

    std::vector<uint8_t> m_vec;
};

struct vcl_allocator_malloc {
    static uint8_t* vcl_allocate(uint64_t size) {
        return reinterpret_cast<uint8_t*>(malloc(size));
    }

    static void vcl_deallocate(uint8_t* ptr) {
        free(ptr);
    }
};

NetworkDescription VCLCompilerImpl::compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const {
    _logger.debug("compile start");

    const auto maxOpsetVersion = _compilerProperties.supportedOpsets;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ze_graph_compiler_version_info_t compilerVersion;
    compilerVersion.major = _compilerProperties.version.major;
    compilerVersion.minor = _compilerProperties.version.minor;
    auto serializedIR = intel_npu::driver_compiler_utils::serializeIR(model, compilerVersion, maxOpsetVersion);

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    _logger.debug("create build flags");
    buildFlags += intel_npu::driver_compiler_utils::serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += intel_npu::driver_compiler_utils::serializeConfig(config, compilerVersion);
    _logger.debug("final build flags to compiler: %s", buildFlags.c_str());
    vcl_executable_desc_t exeDesc = {serializedIR.second.get(),
                                     serializedIR.first,
                                     buildFlags.c_str(),
                                     buildFlags.size()};
    _logger.debug("compiler vcl version: %d.%d", _vclVersion.major, _vclVersion.minor);
    if (_vclVersion.major >= 7 && _vclVersion.minor >= 5) {
        // For VCL 7.5 and later, we can use vclAllocatedExecutableCreate2
        _logger.debug("Using vclAllocatedExecutableCreate2 for 7.5 <= VCL");
        vcl_allocator_vector allocator;
        uint8_t* blob = nullptr;
        size_t size = 0;
        NetworkMetadata metadata;

        THROW_ON_FAIL_FOR_VCL(
            "vclAllocatedExecutableCreate3",
            vclAllocatedExecutableCreate3(_compilerHandle, exeDesc, &allocator, &blob, &size, &metadata),
            _logHandle);
        if (size == 0 || blob == nullptr) {
            OPENVINO_THROW("Failed to create VCL executable, size is zero or blob is null");
        }

        _logger.debug("compile end, blob size:%d", allocator.m_vec.size());
        return NetworkDescription(std::move(allocator.m_vec), std::move(metadata));
    } else if (_vclVersion.major >= 7 && _vclVersion.minor >= 4) {
        // For VCL 7.4 and later, we can use vclAllocatedExecutableCreate2
        _logger.debug("Using vclAllocatedExecutableCreate2 for 7.4 <= VCL < 7.5");
        vcl_allocator_vector allocator;
        uint8_t* blob = nullptr;
        size_t size = 0;

        THROW_ON_FAIL_FOR_VCL("vclAllocatedExecutableCreate2",
                              vclAllocatedExecutableCreate2(_compilerHandle, exeDesc, &allocator, &blob, &size),
                              _logHandle);
        if (size == 0 || blob == nullptr) {
            OPENVINO_THROW("Failed to create VCL executable, size is zero or blob is null");
        }

        // Use empty metadata as VCL does not support metadata extraction
        NetworkMetadata metadata;

        _logger.debug("compile end, blob size:%d", allocator.m_vec.size());
        return NetworkDescription(std::move(allocator.m_vec), std::move(metadata));
    } else if (_vclVersion.major >= 6 && _vclVersion.minor >= 1) {
        // For older versions, we use vclAllocatedExecutableCreate
        _logger.debug("Using vclAllocatedExecutableCreate for 6.1 < VCL < 7.4");

        vcl_allocator_t allocator;
        allocator.allocate = vcl_allocator_malloc::vcl_allocate;
        allocator.deallocate = vcl_allocator_malloc::vcl_deallocate;
        uint8_t* blob = nullptr;
        size_t size = 0;
        THROW_ON_FAIL_FOR_VCL("vclAllocatedExecutableCreate",
                              vclAllocatedExecutableCreate(_compilerHandle, exeDesc, &allocator, &blob, &size),
                              _logHandle);
        if (size == 0 || blob == nullptr) {
            OPENVINO_THROW("Failed to create VCL executable, size is zero or blob is null");
        }

        std::vector<uint8_t> compiledNetwork(blob, blob + size);
        allocator.deallocate(blob);

        // Use empty metadata as VCL does not support metadata extraction
        NetworkMetadata metadata;

        _logger.debug("compile end, blob size:%d", compiledNetwork.size());
        return NetworkDescription(std::move(compiledNetwork), std::move(metadata));
    } else {
        // For versions before 6.1, we use vclExecutableCreate
        _logger.debug("Using vclExecutableCreate for VCL < 6.1");
        vcl_executable_handle_t exeHandle = nullptr;
        THROW_ON_FAIL_FOR_VCL("vclExecutableCreate",
                              vclExecutableCreate(_compilerHandle, exeDesc, &exeHandle),
                              _logHandle);

        size_t size = 0;
        THROW_ON_FAIL_FOR_VCL("vclExecutableGetSerializableBlob",
                              vclExecutableGetSerializableBlob(exeHandle, nullptr, &size),
                              _logHandle);
        if (size == 0) {
            OPENVINO_THROW("Failed to get VCL executable blob size, size is zero");
        }
        std::vector<uint8_t> compiledNetwork(size);
        THROW_ON_FAIL_FOR_VCL("vclExecutableGetSerializableBlob",
                              vclExecutableGetSerializableBlob(exeHandle, compiledNetwork.data(), &size),
                              _logHandle);

        THROW_ON_FAIL_FOR_VCL("vclExecutableDestroy", vclExecutableDestroy(exeHandle), _logHandle);

        // Use empty metadata as VCL does not support metadata extraction
        NetworkMetadata metadata;

        _logger.debug("compile end, blob size:%d", compiledNetwork.size());
        return NetworkDescription(std::move(compiledNetwork), std::move(metadata));
    }
}

intel_npu::NetworkMetadata VCLCompilerImpl::parse(const std::vector<uint8_t>& network, const Config& config) const {
    _logger.debug("parse start");
    // VCL does not support parse, return empty metadata
    return intel_npu::NetworkMetadata();
}

std::vector<ov::ProfilingInfo> VCLCompilerImpl::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                         const std::vector<uint8_t>& network,
                                                                         const intel_npu::Config& config) const {
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

    // profOutput.data = NULL;
    // THROW_ON_FAIL_FOR_VCL("vclGetDecodedProfilingBuffer", vclGetDecodedProfilingBuffer(profilingHandle,
    // VCL_PROFILING_TASK_LEVEL, &profOutput), logHandle); if (profOutput.data == NULL) {
    //     OPENVINO_THROW("Failed to get VCL profiling task level output");
    // }

    // profOutput.data = NULL;
    // THROW_ON_FAIL_FOR_VCL("vclGetDecodedProfilingBuffer", vclGetDecodedProfilingBuffer(profilingHandle,
    // VCL_PROFILING_RAW, &profOutput),logHandle); if (profOutput.data == NULL) {
    //     OPENVINO_THROW("Failed to get VCL profiling raw output");
    // }

    THROW_ON_FAIL_FOR_VCL("vclProfilingDestroy", vclProfilingDestroy(profilingHandle), logHandle);

    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerInfo);  // Return processed profiling info
}

uint32_t VCLCompilerImpl::get_version() const {
    return ZE_MAKE_VERSION(_compilerProperties.version.major, _compilerProperties.version.minor);
}

ov::SupportedOpsMap VCLCompilerImpl::query(const std::shared_ptr<const ov::Model>& model, const Config& config) const {
    _logger.debug("query start");
    const auto maxOpsetVersion = _compilerProperties.supportedOpsets;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ze_graph_compiler_version_info_t compilerVersion;
    compilerVersion.major = _compilerProperties.version.major;
    compilerVersion.minor = _compilerProperties.version.minor;
    auto serializedIR = intel_npu::driver_compiler_utils::serializeIR(model, compilerVersion, maxOpsetVersion);

    std::string buildFlags;
    buildFlags += intel_npu::driver_compiler_utils::serializeConfig(config, compilerVersion);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    vcl_query_handle_t queryHandle;
    vcl_query_desc_t queryDesc = {serializedIR.second.get(), serializedIR.first, buildFlags.c_str(), buildFlags.size()};
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
    // 1. get size of compiler supported options list
    size_t str_size = 0;
    try {
        THROW_ON_FAIL_FOR_VCL("vclGetCompilerSupportedOptions",
                              vclGetCompilerSupportedOptions(_compilerHandle, nullptr, &str_size),
                              _logHandle);

        if (str_size > 0) {
            _logger.debug("obtain list");
            // 2. allocate buffer for it
            options.resize(str_size);
            // 3. populate char list
            THROW_ON_FAIL_FOR_VCL("vclGetCompilerSupportedOptions",
                                  vclGetCompilerSupportedOptions(_compilerHandle, options.data(), &str_size),
                                  _logHandle);

            _logger.debug("Option list size %d, got option list", str_size);
            return true;
        } else {
            _logger.debug("Option list size 0 - skipping!");
        }
    } catch (const std::exception& e) {
        // The API is only supported in new version, just add log here
        _logger.debug("Exception in get_supported_options: %s", e.what());
    }
    _logger.debug("get_supported_options end, no options found");
    return false;
}

bool VCLCompilerImpl::is_option_supported(const std::string& option) const {
    try {
        const char* optname_ch = option.c_str();
        _logger.debug("is_option_supported start for option: %s", optname_ch);
        THROW_ON_FAIL_FOR_VCL("vclGetCompilerIsOptionSupported",
                              vclGetCompilerIsOptionSupported(_compilerHandle, optname_ch, nullptr),
                              _logHandle);
        return true;
    } catch (const std::exception& e) {
        // The API is only supported in new version, just add log here
        _logger.debug("Exception in is_option_supported: %s", e.what());
    }
    _logger.debug("option: %s is not supported", option.c_str());
    return false;
}

}  // namespace intel_npu
