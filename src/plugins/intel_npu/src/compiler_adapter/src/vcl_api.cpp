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
        _logger.debug("Failed to get %s from npu_vcl_compiler", #vcl_symbol);                              \
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
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_version_info_t compilerVersion;
    vcl_version_info_t profilingVersion;
    ret = vclGetVersion(&compilerVersion, &profilingVersion);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to get VCL version: 0x", ret);
    }
    _logger.info("Plugin VCL API Version: %d.%d", VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR);
    _logger.info("Plugin VCL Profiling API Version: %d.%d", VCL_PROFILING_VERSION_MAJOR, VCL_PROFILING_VERSION_MINOR);
    _logger.info("Lib VCL Compiler Version: %d.%d", compilerVersion.major, compilerVersion.minor);
    _logger.info("Lib VCL Profiling Version: %d.%d", profilingVersion.major, profilingVersion.minor);
    _logger.info("Use Lib VCL version to create compiler");

    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version = compilerVersion;
    compilerDesc.debugLevel = static_cast<__vcl_log_level_t>(static_cast<int>(Logger::global().level()) - 1);
    vcl_device_desc_t device_desc;
    device_desc.deviceID = 0x643E;  // Value from intel_npu/src/backend/src/zero_device.cpp
    device_desc.revision = -1;      // -1 to skip the config
    device_desc.tileCount = 1;      // 1 as init value

    if (compilerDesc.debugLevel > VCL_LOG_NONE) {
        ret = vclCompilerCreate(&compilerDesc, &device_desc, &_compilerHandle, &_logHandle);
    } else {
        ret = vclCompilerCreate(&compilerDesc, &device_desc, &_compilerHandle, nullptr);
    }

    ret = vclCompilerGetProperties(_compilerHandle, &_compilerProperties);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create VCL compiler: 0x", ret);
    }
    _logger.info("VCL Compiler created successfully");
    _logger.info("VCL Compiler Properties: ID: %s, Version: %d.%d, Supported Opsets: %u",
                 _compilerProperties.id,
                 _compilerProperties.version.major,
                 _compilerProperties.version.minor,
                 _compilerProperties.supportedOpsets);
}

VCLCompilerImpl::~VCLCompilerImpl() {
    if (_compilerHandle) {
        vcl_result_t ret = vclCompilerDestroy(_compilerHandle);
        if (ret != VCL_RESULT_SUCCESS) {
            _logger.error("Failed to destroy VCL compiler: 0x%x", ret);
        }
    }
    if (_logHandle) {
        _logHandle = nullptr;  // Log handle is released automatically with the compiler
    }
    _logger.info("VCL Compiler destroyed successfully");
}

NetworkDescription VCLCompilerImpl::compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const {
    _logger.debug("compile start");
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    const auto maxOpsetVersion = _compilerProperties.supportedOpsets;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ze_graph_compiler_version_info_t compilerVersion;
    compilerVersion.major = _compilerProperties.version.major;
    compilerVersion.minor = _compilerProperties.version.minor;
    auto serializedIR = intel_npu::driver_compiler_utils::serializeIR(model, compilerVersion, maxOpsetVersion);

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    _logger.debug("build flags");
    buildFlags += intel_npu::driver_compiler_utils::serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += intel_npu::driver_compiler_utils::serializeConfig(config, compilerVersion);

    vcl_executable_desc_t exeDesc = {serializedIR.second.get(),
                                     serializedIR.first,
                                     buildFlags.c_str(),
                                     buildFlags.size()};
    vcl_executable_handle_t exeHandle = nullptr;
    ret = vclExecutableCreate(_compilerHandle, exeDesc, &exeHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create VCL executable: 0x", ret);
    }
    size_t size = 0;
    ret = vclExecutableGetSerializableBlob(exeHandle, nullptr, &size);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to get VCL executable blob size: 0x", ret);
    }
    std::vector<uint8_t> compiledNetwork(size);
    ret = vclExecutableGetSerializableBlob(exeHandle, compiledNetwork.data(), &size);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to get VCL executable blob: 0x", ret);
    }
    ret = vclExecutableDestroy(exeHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to destroy VCL executable: 0x", ret);
    }

    // Use empty metadata as VCL does not support metadata extraction
    NetworkMetadata metadata;

    _logger.debug("compile end, blob size:%d", compiledNetwork.size());
    return NetworkDescription(std::move(compiledNetwork), std::move(metadata));
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
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_profiling_handle_t profilingHandle;
    vcl_profiling_input_t profilingInput = {network.data(), network.size(), profData.data(), profData.size()};
    vcl_log_handle_t logHandle;
    ret = vclProfilingCreate(&profilingInput, &profilingHandle, &logHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create VCL profiling handler: 0x", ret);
    }

    vcl_profiling_properties_t profProperties;
    ret = vclProfilingGetProperties(profilingHandle, &profProperties);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to get VCL profiling properties: 0x", ret);
    }
    _logger.info("VCL Profiling Properties: Version: %d.%d",
                 profProperties.version.major,
                 profProperties.version.minor);

    // We only use layer level info
    vcl_profiling_request_type_t request = VCL_PROFILING_LAYER_LEVEL;

    vcl_profiling_output_t profOutput;
    profOutput.data = NULL;
    ret = vclGetDecodedProfilingBuffer(profilingHandle, request, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
        OPENVINO_THROW("Failed to get VCL profiling output: 0x", ret);
    }

    std::vector<ze_profiling_layer_info> layerInfo(profOutput.size / sizeof(ze_profiling_layer_info));
    if (profOutput.size > 0) {
        _logger.debug("VCL profiling output size: %d", profOutput.size);
        std::memcpy(layerInfo.data(), profOutput.data, profOutput.size);
    }

    // profOutput.data = NULL;
    // ret = vclGetDecodedProfilingBuffer(profilingHandle, VCL_PROFILING_TASK_LEVEL, &profOutput);
    // if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
    //     OPENVINO_THROW("Failed to get VCL profiling task level output: 0x", ret);
    // }

    // profOutput.data = NULL;
    // ret = vclGetDecodedProfilingBuffer(profilingHandle, VCL_PROFILING_RAW, &profOutput);
    // if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
    //     OPENVINO_THROW("Failed to get VCL profiling raw output: 0x", ret);
    // }

    ret = vclProfilingDestroy(profilingHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to destroy VCL profiling handler: 0x", ret);
    }

    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerInfo);  // Return processed profiling info
}

uint32_t VCLCompilerImpl::get_version() const {
    return ZE_MAKE_VERSION(_compilerProperties.version.major, _compilerProperties.version.minor);
}

ov::SupportedOpsMap VCLCompilerImpl::query(const std::shared_ptr<const ov::Model>& model, const Config& config) const {
    _logger.debug("query start");
    vcl_result_t ret = VCL_RESULT_SUCCESS;
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
    ret = vclQueryNetworkCreate(_compilerHandle, queryDesc, &queryHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create VCL query network: 0x", ret);
    }

    uint64_t size = 0;
    ret = vclQueryNetwork(queryHandle, nullptr, &size);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to query network size: 0x", ret);
    }

    std::vector<char> supportedLayers(size);
    ret = vclQueryNetwork(queryHandle, reinterpret_cast<uint8_t*>(supportedLayers.data()), &size);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to query network: 0x", ret);
    }

    ret = vclQueryNetworkDestroy(queryHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to destroy VCL query network: 0x", ret);
    }

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
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    // 1. get size of compiler supported options list
    size_t str_size = 0;
    try {
        ret = vclGetCompilerSupportedOptions(_compilerHandle, nullptr, &str_size);

        if (ret != VCL_RESULT_SUCCESS) {
            _logger.debug("Failed to get size of option list %x", ret);
            return false;
        }

        if (str_size > 0) {
            _logger.debug("obtain list");
            // 2. allocate buffer for it
            options.resize(str_size);
            // 3. populate char list
            ret = vclGetCompilerSupportedOptions(_compilerHandle, options.data(), &str_size);

            if (ret == VCL_RESULT_SUCCESS) {
                _logger.debug("pfnCompilerGetSupportedOptions - list size %d, got option list", str_size);
                return true;
            } else {
                _logger.debug("Failed to get content of option list 0x%x", ret);
            }
        } else {
            _logger.debug("pfnCompilerGetSupportedOptions - list size 0 - skipping!");
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
        auto result = vclGetCompilerIsOptionSupported(_compilerHandle, optname_ch, nullptr);
        if (result == VCL_RESULT_SUCCESS) {
            return true;
        }
    } catch (const std::exception& e) {
        // The API is only supported in new version, just add log here
        _logger.debug("Exception in is_option_supported: %s", e.what());
    }
    return false;
}

}  // namespace intel_npu
