// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_impl.hpp"

#include <mutex>

#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/profiling.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "vcl_serializer.hpp"
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

bool isUseBaseModelSerializer(UsedVersion useVersion, const intel_npu::FilteredConfig& config) {
    // vcl serializer(No copy) is only support for vcl version >= 7.5
    if (useVersion.Major < 7 || (useVersion.Major == 7 && useVersion.Minor < 5)) {
        return true;
    }

    // user pass use_base_model_serializer config
    if (config.isAvailable(ov::intel_npu::use_base_model_serializer.name()) &&
        config.has(ov::intel_npu::use_base_model_serializer.name())) {
        return config.get<intel_npu::USE_BASE_MODEL_SERIALIZER>();
    }

    // user pass model_serializer_version config
    if (config.isAvailable(ov::intel_npu::model_serializer_version.name()) &&
        config.has(ov::intel_npu::model_serializer_version.name())) {
        return (config.get<intel_npu::MODEL_SERIALIZER_VERSION>() ==
                ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY);
    }

    // No VCL serializer was chosen explicitly, will default to the "no weights copy" implementation
    return false;
}

}  // namespace

namespace intel_npu {

// clang-format off
#define vcl_symbols_list()                                  \
    vcl_symbol_statement(vclGetVersion)                     \
    vcl_symbol_statement(vclCompilerCreate)                 \
    vcl_symbol_statement(vclCompilerDestroy)                \
    vcl_symbol_statement(vclCompilerGetProperties)          \
    vcl_symbol_statement(vclQueryNetworkCreate)             \
    vcl_symbol_statement(vclQueryNetwork)                   \
    vcl_symbol_statement(vclQueryNetworkDestroy)            \
    vcl_symbol_statement(vclExecutableCreate)               \
    vcl_symbol_statement(vclAllocatedExecutableCreate)      \
    vcl_symbol_statement(vclExecutableDestroy)              \
    vcl_symbol_statement(vclExecutableGetSerializableBlob)  \
    vcl_symbol_statement(vclProfilingCreate)                \
    vcl_symbol_statement(vclGetDecodedProfilingBuffer)      \
    vcl_symbol_statement(vclProfilingDestroy)               \
    vcl_symbol_statement(vclProfilingGetProperties)         \
    vcl_symbol_statement(vclLogHandleGetString)             \
    vcl_symbol_statement(vclAllocatedExecutableCreate2)     \
    vcl_symbol_statement(vclGetCompilerSupportedOptions)    \
    vcl_symbol_statement(vclGetCompilerIsOptionSupported)   \


// symbols that may not be supported in older versions of vcl
#define vcl_weak_symbols_list()                             \
    vcl_symbol_statement(vclAllocatedExecutableCreateWSOneShot)
// clang-format on

class VCLApi {
public:
    VCLApi();
    VCLApi(const VCLApi& other) = delete;
    VCLApi(VCLApi&& other) = delete;
    void operator=(const VCLApi&) = delete;
    void operator=(VCLApi&&) = delete;

    static const std::shared_ptr<VCLApi> getInstance();
    std::shared_ptr<void> getLibrary() const {
        return lib;
    }

#define vcl_symbol_statement(vcl_symbol) decltype(&::vcl_symbol) vcl_symbol;
    vcl_symbols_list();
    vcl_weak_symbols_list();
#undef vcl_symbol_statement

private:
    std::shared_ptr<void> lib;
    Logger _logger;
};

#define vcl_symbol_statement(vcl_symbol)                                                                            \
    template <typename... Args>                                                                                     \
    inline typename std::invoke_result<decltype(&::vcl_symbol), Args...>::type wrapped_##vcl_symbol(Args... args) { \
        const auto& ptr = VCLApi::getInstance();                                                                    \
        if (ptr->vcl_symbol == nullptr) {                                                                           \
            OPENVINO_THROW("Unsupported vcl_symbol " #vcl_symbol);                                                  \
        }                                                                                                           \
        return ptr->vcl_symbol(std::forward<Args>(args)...);                                                        \
    }
vcl_symbols_list();
vcl_weak_symbols_list();
#undef vcl_symbol_statement
#define vcl_symbol_statement(vcl_symbol) inline decltype(&::vcl_symbol) vcl_symbol = wrapped_##vcl_symbol;
vcl_symbols_list();
vcl_weak_symbols_list();
#undef vcl_symbol_statement

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

VCLApi::VCLApi() : _logger("VCLApi", Logger::global().level()) {
    const std::filesystem::path baseName = "openvino_intel_npu_compiler";
    try {
        auto libpath = ov::util::make_plugin_library_name({}, baseName);
        _logger.debug("Try to load openvino_intel_npu_compiler");
        this->lib = ov::util::load_shared_object(libpath);
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to load openvino_intel_npu_compiler");
        OPENVINO_THROW(error.what());
    }

    try {
#define vcl_symbol_statement(vcl_symbol) \
    this->vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol));
        vcl_symbols_list();
#undef vcl_symbol_statement
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to get formal symbols from openvino_intel_npu_compiler");
        OPENVINO_THROW(error.what());
    }

#define vcl_symbol_statement(vcl_symbol)                                                                      \
    try {                                                                                                     \
        this->vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol)); \
    } catch (const std::runtime_error&) {                                                                     \
        _logger.debug("Failed to get %s from openvino_intel_npu_compiler", #vcl_symbol);                      \
        this->vcl_symbol = nullptr;                                                                           \
    }
    vcl_weak_symbols_list();
#undef vcl_symbol_statement

#define vcl_symbol_statement(vcl_symbol) vcl_symbol = this->vcl_symbol;
    vcl_symbols_list();
    vcl_weak_symbols_list();
#undef vcl_symbol_statement
}

const std::shared_ptr<VCLApi> VCLApi::getInstance() {
    static std::shared_ptr<VCLApi> instance = std::make_shared<VCLApi>();
    return instance;
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
    compilerDesc.debugLevel = static_cast<__vcl_log_level_t>(static_cast<int>(Logger::global().level()) - 1);

    // This information cannot be determined during the initialization phase; set device desc default value, the related
    // info will be processed in compile phase if passed by user.
    _logger.info("Device description is not provided, using default values");
    vcl_device_desc_t device_desc = {sizeof(vcl_device_desc_t),
                                     0x00,
                                     static_cast<uint16_t>(-1),
                                     static_cast<uint16_t>(-1)};
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

std::shared_ptr<void> VCLCompilerImpl::getLinkedLibrary() const {
    return VCLApi::getInstance()->getLibrary();
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

struct vcl_allocator_vector_2 : vcl_allocator2_t {
    vcl_allocator_vector_2() : vcl_allocator2_t{vector_allocate, vector_deallocate} {}

    static uint8_t* vector_allocate(vcl_allocator2_t* allocator, size_t size) {
        vcl_allocator_vector_2* vecAllocator = static_cast<vcl_allocator_vector_2*>(allocator);
        auto newVec = std::make_shared<std::vector<uint8_t>>();
        newVec->resize(size);
        uint8_t* ptr = newVec->data();
        vecAllocator->m_vector.emplace_back(newVec);
        return ptr;
    }

    static void vector_deallocate(vcl_allocator2_t* allocator, uint8_t* ptr) {
        vcl_allocator_vector_2* vecAllocator = static_cast<vcl_allocator_vector_2*>(allocator);
        vecAllocator->m_vector.clear();
        vecAllocator->m_vector.shrink_to_fit();
    }

    std::vector<std::shared_ptr<std::vector<uint8_t>>> m_vector;
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
    return compile(model, config, false);
}

NetworkDescription VCLCompilerImpl::compile(const std::shared_ptr<const ov::Model>& model,
                                            const Config& config,
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

    const FilteredConfig* filteredConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (filteredConfig == nullptr) {
        OPENVINO_THROW("config is not FilteredConfig");
    }
    FilteredConfig updatedConfig = *filteredConfig;
    bool useBaseModelSerializer = true;
    useBaseModelSerializer = isUseBaseModelSerializer(usedVersion, updatedConfig);
    _logger.debug("serialize IR method is %s",
                  useBaseModelSerializer ? "base vcl serializer" : "vcl serializer (not copy weights)");
    auto serializedIR = driver_compiler_utils::serializeIR(model,
                                                           compilerVersion,
                                                           maxOpsetVersion,
                                                           useBaseModelSerializer,
                                                           false,
                                                           storeWeightlessCacheAttributeFlag);

    std::string buildFlags;
    _logger.debug("create build flags");
    buildFlags += driver_compiler_utils::serializeIOInfo(model, true);
    buildFlags += " ";
    buildFlags += driver_compiler_utils::serializeConfig(updatedConfig, compilerVersion);
    _logger.debug("final build flags to compiler: %s", buildFlags.c_str());

    vcl_executable_desc_t exeDesc = {serializedIR.buffer.get(),
                                     serializedIR.size,
                                     buildFlags.c_str(),
                                     buildFlags.size()};

    if (usedVersion.Major >= 7 && usedVersion.Minor >= 4) {
        // support the lastest vcl api
        // For VCL 7.4 and later, we can use vclAllocatedExecutableCreate2
        _logger.debug("Using vclAllocatedExecutableCreate2 for 7.4 <= VCL");
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
    } else {
        OPENVINO_THROW("Not supported VCL version: %d.%d, please use VCL 6.1 or later",
                       _vclVersion.major,
                       _vclVersion.minor);
    }
}

std::vector<std::shared_ptr<NetworkDescription>> VCLCompilerImpl::compileWsOneShot(
    const std::shared_ptr<ov::Model>& model,
    const Config& config) const {
    _logger.debug("compileWsOneShot start");

    const auto maxOpsetVersion = _compilerProperties.supportedOpsets;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ze_graph_compiler_version_info_t compilerVersion;
    compilerVersion.major = _compilerProperties.version.major;
    compilerVersion.minor = _compilerProperties.version.minor;

    const FilteredConfig* filteredConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (filteredConfig == nullptr) {
        OPENVINO_THROW("config is not FilteredConfig");
    }
    FilteredConfig updatedConfig = *filteredConfig;
    bool useBaseModelSerializer = true;
    useBaseModelSerializer = isUseBaseModelSerializer({7, 5}, updatedConfig);
    _logger.debug("serialize IR method is %s",
                  useBaseModelSerializer ? "base vcl serializer" : "vcl serializer (not copy weights)");
    auto serializedIR = driver_compiler_utils::serializeIR(model,
                                                           compilerVersion,
                                                           maxOpsetVersion,
                                                           useBaseModelSerializer,
                                                           false,
                                                           true);

    std::string buildFlags;
    _logger.debug("create build flags");
    buildFlags += driver_compiler_utils::serializeIOInfo(model, true);
    buildFlags += " ";
    buildFlags += driver_compiler_utils::serializeConfig(updatedConfig, compilerVersion);
    _logger.debug("final build flags to compiler: %s", buildFlags.c_str());

    vcl_executable_desc_t exeDesc = {serializedIR.buffer.get(),
                                     serializedIR.size,
                                     buildFlags.c_str(),
                                     buildFlags.size()};
    _logger.debug("compiler vcl version: %d.%d", _vclVersion.major, _vclVersion.minor);

    _logger.debug("Using vclAllocatedExecutableCreateWSOneShot");
    vcl_allocator_vector_2 allocator;

    THROW_ON_FAIL_FOR_VCL("vclAllocatedExecutableCreateWSOneShot",
                          vclAllocatedExecutableCreateWSOneShot(_compilerHandle, exeDesc, &allocator),
                          _logHandle);

    if (allocator.m_vector.size() == 0) {
        OPENVINO_THROW("Failed to create VCL executable, blobCount is zero");
    }

    std::vector<std::shared_ptr<NetworkDescription>> networkDescrs;
    for (auto& blob : allocator.m_vector) {
        // Use empty metadata as VCL does not support metadata extraction
        NetworkMetadata metadata;
        networkDescrs.emplace_back(std::make_shared<NetworkDescription>(std::move(*blob), std::move(metadata)));
    }
    return networkDescrs;
}

NetworkDescription VCLCompilerImpl::compileWsIterative(const std::shared_ptr<ov::Model>& model,
                                                       const Config& config,
                                                       size_t callNumber) const {
    _logger.debug("compileWsIterative start");
    const FilteredConfig* filteredConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (filteredConfig == nullptr) {
        OPENVINO_THROW("config is not FilteredConfig");
    }
    FilteredConfig updatedConfig = *filteredConfig;
    updatedConfig.update({{ov::intel_npu::ws_compile_call_number.name(), std::to_string(callNumber)}});
    return compile(model, updatedConfig, true);
}

intel_npu::NetworkMetadata VCLCompilerImpl::parse(const std::vector<uint8_t>& network, const Config& config) const {
    // VCL returns empty metadata. In plugin adapter, use driver metadata instead.
    OPENVINO_THROW_NOT_IMPLEMENTED("VCL does not support parse.");
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

    THROW_ON_FAIL_FOR_VCL("vclProfilingDestroy", vclProfilingDestroy(profilingHandle), logHandle);

    // Return processed profiling info
    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerInfo);
}

uint32_t VCLCompilerImpl::get_version() const {
    return ZE_MAKE_VERSION(_compilerProperties.version.major, _compilerProperties.version.minor);
}

ov::SupportedOpsMap VCLCompilerImpl::query(const std::shared_ptr<const ov::Model>& model, const Config& config) const {
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
    const FilteredConfig* filteredConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (filteredConfig == nullptr) {
        OPENVINO_THROW("config is not FilteredConfig");
    }
    FilteredConfig updatedConfig = *filteredConfig;
    bool useBaseModelSerializer = true;
    useBaseModelSerializer = isUseBaseModelSerializer(usedVersion, updatedConfig);
    _logger.debug("serialize IR method is %s",
                  useBaseModelSerializer ? "base vcl serializer" : "vcl serializer (not copy weights)");
    auto serializedIR =
        driver_compiler_utils::serializeIR(model, compilerVersion, maxOpsetVersion, useBaseModelSerializer);

    std::string buildFlags;
    buildFlags += driver_compiler_utils::serializeConfig(updatedConfig, compilerVersion);
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

bool VCLCompilerImpl::is_option_supported(const std::string& option, std::optional<std::string> optValue) const {
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

}  // namespace intel_npu
