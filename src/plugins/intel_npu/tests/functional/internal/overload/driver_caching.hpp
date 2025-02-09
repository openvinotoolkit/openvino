#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <iostream>

#include <common_test_utils/test_assertions.hpp>
#include <openvino/opsets/opset1.hpp>

#include "base/ov_behavior_test_utils.hpp"

#include <openvino/runtime/core.hpp>
#include "openvino/core/except.hpp"

#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"

#include "ir_serializer.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

#include <iostream>


namespace ov {
namespace test {
namespace behavior {

bool containsCacheStatus(const std::string& str, const std::string cmpstr); 

inline std::shared_ptr<ov::Model> getConstantGraph() {
    auto now = std::chrono::system_clock::now();
    auto timeStamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add" + std::to_string(timeStamp));
    return std::make_shared<ov::Model>(ov::OutputVector{add->output(0)}, ov::ParameterVector{param});
}

bool containsCacheStatus(const std::string& str, const std::string cmpstr) {
    return str.find(cmpstr) != std::string::npos;
}

/**
 * @brief A standard copy function concerning memory segments. Additional checks on the given arguments are performed
 * before copying.
 * @details This is meant as a replacement for the legacy "ie_memcpy" function coming from the OpenVINO API.
 */
void checkedMemcpy(void* destination, size_t destinationSize, void const* source, size_t numberOfBytes) {
    if (numberOfBytes == 0) {
        return;
    }

    OPENVINO_ASSERT(destination != nullptr, "Memcpy: received a null destination address");
    OPENVINO_ASSERT(source != nullptr, "Memcpy: received a null source address");
    OPENVINO_ASSERT(numberOfBytes <= destinationSize,
                    "Memcpy: the source buffer does not fit inside the destination one");
    OPENVINO_ASSERT(numberOfBytes <= (destination > source ? ((uintptr_t)destination - (uintptr_t)source)
                                                           : ((uintptr_t)source - (uintptr_t)destination)),
                    "Memcpy: the offset between the two buffers does not allow a safe execution of the operation");

    memcpy(destination, source, numberOfBytes);
}


// print ze_structure_type_graph_ext_t
const char* getStructureTypeString(ze_structure_type_graph_ext_t stype)
{
    switch (stype)
    {
    case ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES: return "ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES";
    case ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES: return "ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES";
    case ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES: return "ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES";
    case ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES: return "ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_GRAPH_ACTIVATION_KERNEL: return "ZE_STRUCTURE_TYPE_GRAPH_ACTIVATION_KERNEL";
    case ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA: return "ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA";
    case ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC_DEPRECATED: return "ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC_DEPRECATED";
    case ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_PROFILING_QUERY_EXP_DESC: return "ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_PROFILING_QUERY_EXP_DESC";
    default: return "Unknown";
    }
}

// print ze_graph_init_stage_t
const char* getInitStageString(ze_graph_init_stage_t stage)
{
    switch (stage)
    {
    case ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE: return "ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE";
    case ZE_GRAPH_STAGE_INITIALIZE: return "ZE_GRAPH_STAGE_INITIALIZE";
    case ZE_GRAPH_STAGE_FORCE_UINT32: return "ZE_GRAPH_STAGE_FORCE_UINT32";
    default: return "Unknown";
    }
}

// print ze_graph_properties_3_t
void printGraphProperties(const ze_graph_properties_3_t& graphProperties)
{
    std::cout << "stype: " << getStructureTypeString(graphProperties.stype) << std::endl;
    std::cout << "pNext: " << graphProperties.pNext << std::endl;
    std::cout << "numGraphArgs: " << graphProperties.numGraphArgs << std::endl;
    std::cout << "initStageRequired: " << getInitStageString(graphProperties.initStageRequired) << std::endl;
    std::cout << "flags: " << graphProperties.flags << std::endl;
}

typedef std::tuple<std::string,  // Device name
                   ov::AnyMap    // Config
                   >
    CompileAndModelCachingParams;

using SerializedIR = std::pair<size_t, std::shared_ptr<uint8_t>>;
using IRSerializer = ::intel_npu::driver_compiler_utils::IRSerializer;
class CompileAndDriverCaching : public testing::WithParamInterface<CompileAndModelCachingParams>,
                                public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndModelCachingParams> obj) {
        std::string targetDevice;
        ov::AnyMap m_configuration;
        std::tie(targetDevice, m_configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!m_configuration.empty()) {
            for (auto& configItem : m_configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, m_configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        m_initStruct = std::make_shared<::intel_npu::ZeroInitStructsHolder>();
        if (!m_initStruct) {
            GTEST_SKIP() << "ZeroInitStructsHolder init failed, ZeroInitStructsHolder is a nullptr";
        }

        ze_graph_dditable_ext_decorator& graph_ddi_table_ext = m_initStruct->getGraphDdiTable();
        uint32_t graphDdiExtVersion = graph_ddi_table_ext.version();
        if (graphDdiExtVersion < ZE_GRAPH_EXT_VERSION_1_5) {
            GTEST_SKIP() << "Skipping test for Driver version less than 1.5, current driver version: "
                         << graphDdiExtVersion;
        }

        APIBaseTest::SetUp();
    }

    void TearDown() override {
        if (!m_cachedir.empty()) {
            ov::test::utils::removeFilesWithExt(m_cachedir, "blob");
            ov::test::utils::removeDir(m_cachedir);
        }
        ov::test::utils::PluginCache::get().reset();
        APIBaseTest::TearDown();
    }

    SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                            ze_graph_compiler_version_info_t compilerVersion,
                            const uint32_t supportedOpsetVersion) const;


protected:
    std::shared_ptr<ov::Core> m_core = utils::PluginCache::get().core();
    ov::AnyMap m_configuration;
    std::shared_ptr<ov::Model> m_function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> m_initStruct;
    std::string m_cachedir;
};

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        // The value of cache_encryption_callbacks cannot be converted to std::string
        if (value.first == ov::cache_encryption_callbacks.name()) {
            continue;
        }
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

::intel_npu::Config merge_configs(const ::intel_npu::Config& globalConfig,
                            const std::map<std::string, std::string>& rawConfig,
                            ::intel_npu::OptionMode mode = ::intel_npu::OptionMode::Both) {
    ::intel_npu::Config localConfig = globalConfig;
    localConfig.update(rawConfig, mode);
    return localConfig;
}

SerializedIR CompileAndDriverCaching::serializeIR(const std::shared_ptr<const ov::Model>& model,
                                                ze_graph_compiler_version_info_t compilerVersion,
                                                const uint32_t supportedOpsetVersion) const {
    IRSerializer irSerializer(model, supportedOpsetVersion);

    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(irSerializer.getXmlSize());
    const uint64_t weightsSize = static_cast<uint64_t>(irSerializer.getWeightsSize());

    OPENVINO_ASSERT(numberOfInputData < maxNumberOfElements);
    if (xmlSize >= maxSizeOfXML) {
        OPENVINO_THROW("Xml file is too big to process. xmlSize: ", xmlSize, " >= maxSizeOfXML: ", maxSizeOfXML);
    }
    if (weightsSize >= maxSizeOfWeights) {
        OPENVINO_THROW("Bin file is too big to process. xmlSize: ",
                       weightsSize,
                       " >= maxSizeOfWeights: ",
                       maxSizeOfWeights);
    }

    const uint64_t sizeOfSerializedIR = sizeof(compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xmlSize + sizeof(weightsSize) + weightsSize;

    // use array to avoid vector's memory zeroing overhead
    std::shared_ptr<uint8_t> buffer(new uint8_t[sizeOfSerializedIR], std::default_delete<uint8_t[]>());
    uint8_t* serializedIR = buffer.get();

    uint64_t offset = 0;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &compilerVersion, sizeof(compilerVersion));
    offset += sizeof(compilerVersion);

    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    // xml data is filled in serializeModel()
    uint64_t xmlOffset = offset;
    offset += xmlSize;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    // weights data is filled in serializeModel()
    uint64_t weightsOffset = offset;
    offset += weightsSize;

    irSerializer.serializeModelToBuffer(serializedIR + xmlOffset, serializedIR + weightsOffset);

    OPENVINO_ASSERT(offset == sizeOfSerializedIR);

    return std::make_pair(sizeOfSerializedIR, buffer);
}



TEST_P(CompileAndDriverCaching, CompilationCache) {
    // ze_graph_dditable_ext_decorator& graph_ddi_table_ext = m_initStruct->getGraphDdiTable();// seems no usful

    auto options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config ConfigInfo(options);
    const std::map<std::string, std::string> localPropertiesMap = any_copy(m_configuration);
    auto localConfig = merge_configs(ConfigInfo, localPropertiesMap);

    uint32_t flags = ZE_GRAPH_FLAG_NONE;
    const auto set_cache_dir = localConfig.get<::intel_npu::CACHE_DIR>();
    if (!set_cache_dir.empty() || localConfig.get<::intel_npu::BYPASS_UMD_CACHING>()) {
        flags = flags | ZE_GRAPH_FLAG_DISABLE_CACHING;
    }

    m_function = getConstantGraph();
    auto compilerProperties = m_initStruct->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;

    auto serializedIR = serializeIR(m_function, compilerVersion, maxOpsetVersion);
    // auto serializedIR = DriverCompilerAdapter::serializeIR(m_function, compilerVersion, maxOpsetVersion);

    std::string buildFlags = "--inputs_precisions=\"A:fp16 B:fp16 C:fp16\" --inputs_layouts=\"A:C B:C C:C\" --outputs_precisions=\"Y:fp16\" --outputs_layouts=\"Y:C\" --config NPU_PLATFORM=\"4000\" DEVICE_ID=\"NPU.4000\" NPU_COMPILATION_MODE=\"DefaultHW\"";
    // const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));
    // buildFlags += serializeIOInfo(m_function, useIndices);  //可以使用src/plugins/intel_npu/src/compiler_adapter/src/plugin_compiler_adapter.cpp：319中的内容吗
    // buildFlags += " ";
    // buildFlags += serializeConfig(config, compilerVersion);//可以使用src/plugins/intel_npu/src/compiler_adapter/src/plugin_compiler_adapter.cpp：408中的内容吗
    
    
    // buildFlags += DriverCompilerAdapter::serializeIOInfo(m_function, useIndices);  //可以使用src/plugins/intel_npu/src/compiler_adapter/src/plugin_compiler_adapter.cpp：319中的内容吗
    // buildFlags += " ";
    // buildFlags += DriverCompilerAdapter::serializeConfig(config, compilerVersion);//可以使用src/plugins/intel_npu/src/compiler_adapter/src/plugin_compiler_adapter.cpp：408中的内容吗


    ze_graph_handle_t graphHandle;
    // 这个和 ze_graph_dditable_ext_decorator& graph_ddi_table_ext = m_initStruct->getGraphDdiTable();的关系是什么
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            ZE_GRAPH_FORMAT_NGRAPH_LITE,
                            serializedIR.first,
                            serializedIR.second.get(),
                            buildFlags.c_str(),
                            flags};
    ze_graph_build_log_handle_t graphBuildLogHandle = nullptr;

    /// before compile, property check
    ze_graph_properties_3_t graphProperties = {};
    std::cout << "-N1-before ze_graph_properties_3_t init-------start-------------- " << desc.flags << std::endl;
    printGraphProperties(graphProperties);
    std::cout << "-N2-before ze_graph_properties_3_t init-------end-------------- " << desc.flags << std::endl;
    auto result1 = m_initStruct->getGraphDdiTable().pfnGetProperties3(graphHandle, &graphProperties);
    std::cout << "   0) result of _zerom_initStruct->getGraphDdiTable().pfnGetProperties3 is " << uint64_t(result1) << std::endl;
    std::cout << "-N3-before ze_graph_properties_3_t init-------start-------------- " << desc.flags << std::endl;
    printGraphProperties(graphProperties);
    std::cout << "-N4-before ze_graph_properties_3_t init-------end-------------- " << desc.flags << std::endl;
    std::cout << "   1) graphProperties.flags is " << graphProperties.flags << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;


    auto result = m_initStruct->getGraphDdiTable().pfnCreate3(m_initStruct->getContext(),
                                                        m_initStruct->getDevice(),
                                                        &desc,
                                                        &graphHandle,
                                                        &graphBuildLogHandle);
    /// after compile, property check
    std::cout << "-N1-after ze_graph_properties_3_t init-------start-------------- " << desc.flags << std::endl;
    printGraphProperties(graphProperties);
    std::cout << "-N2-after ze_graph_properties_3_t init-------end-------------- " << desc.flags << std::endl;
    auto result2 = m_initStruct->getGraphDdiTable().pfnGetProperties3(graphHandle, &graphProperties);
    std::cout << "-N3-after ze_graph_properties_3_t init-------start-------------- " << desc.flags << std::endl;
    printGraphProperties(graphProperties);
    std::cout << "-N4-after ze_graph_properties_3_t init-------end-------------- " << desc.flags << std::endl;



    std::cout << "   3) result of _zerom_initStruct->getGraphDdiTable().pfnGetProperties3 is " << uint64_t(result2) << std::endl;
    //    ZE_GRAPH_PROPERTIES_FLAG_LOADED_FROM_CACHE = ZE_BIT(0),       ///< graph object is loaded from driver cache
    //    #define ZE_BIT( _i )  ( 1 << _i )

    std::cout << "   4) graphProperties.flags is " << graphProperties.flags << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;

    // ov::CompiledModel execNet;
    // m_function = getConstantGraph();

    // // first compilation
    // auto startFirstCompilationTime = std::chrono::high_resolution_clock::now();
    // OV_ASSERT_NO_THROW(execNet = m_core->compile_model(m_function, target_device, m_configuration));
    // std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    // std::printf("==[1.1][EmptyConfig] driver log content : #%s#\n", firstCompilationDriverLog.c_str());
    // ;
    // EXPECT_TRUE(containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored"));
    // auto endFirstCompilationTime = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> durationFirstCompilation = endFirstCompilationTime - startFirstCompilationTime;

    // // second compilation
    // auto startSecondCompilationTime = std::chrono::high_resolution_clock::now();
    // OV_ASSERT_NO_THROW(execNet = m_core->compile_model(m_function, target_device, m_configuration));
    // auto endSecondCompilationTime = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> durationSecondCompilation = startSecondCompilationTime - endSecondCompilationTime;
    // std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    // std::printf("==[1.2][EmptyConfig] driver log content : #%s#\n", secondCompilationDriverLog.c_str());
    // EXPECT_TRUE(containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));

    // EXPECT_LT(durationSecondCompilation, durationFirstCompilation)
    //     << "The duration of the second compilation should be less than the first due to UMD Caching.";
}

//extension is lower than https://github.com/intel/level-zero-npu-extensions/commit/d16f5d09fd695c1aac0c29524881fec7ccf7d27e will skip this part.
// TEST_P(CompileAndDriverCaching, CompilationCacheWithEmptyConfig) {
//     ze_graph_dditable_ext_decorator& graph_ddi_table_ext = m_initStruct->getGraphDdiTable();

//     ov::CompiledModel execNet;
//     m_function = getConstantGraph();

//     // first compilation
//     auto startFirstCompilationTime = std::chrono::high_resolution_clock::now();
//     OV_ASSERT_NO_THROW(execNet = m_core->compile_model(m_function, target_device, m_configuration));
//     std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("==[1.1][EmptyConfig] driver log content : #%s#\n", firstCompilationDriverLog.c_str());
//     ;
//     EXPECT_TRUE(containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored"));
//     auto endFirstCompilationTime = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> durationFirstCompilation = endFirstCompilationTime - startFirstCompilationTime;

//     // second compilation
//     auto startSecondCompilationTime = std::chrono::high_resolution_clock::now();
//     OV_ASSERT_NO_THROW(execNet = m_core->compile_model(m_function, target_device, m_configuration));
//     auto endSecondCompilationTime = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> durationSecondCompilation = startSecondCompilationTime - endSecondCompilationTime;
//     std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("==[1.2][EmptyConfig] driver log content : #%s#\n", secondCompilationDriverLog.c_str());
//     EXPECT_TRUE(containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));

//     EXPECT_LT(durationSecondCompilation, durationFirstCompilation)
//         << "The duration of the second compilation should be less than the first due to UMD Caching.";
// }

// TEST_P(CompileAndDriverCaching, CompilationCacheWithOVCacheConfig) {
//     ze_graph_dditable_ext_decorator& graph_ddi_table_ext = m_initStruct->getGraphDdiTable();

//     m_configuration[ov::cache_dir.name()] = "./testCacheDir";
//     m_cachedir = m_configuration[ov::cache_dir.name()].as<std::string>();
//     ov::CompiledModel execNet;
//     m_function = getConstantGraph();

//     // first compilation
//     OV_ASSERT_NO_THROW(execNet = m_core->compile_model(m_function, target_device, m_configuration));
//     std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("==[2.1][OVCacheConfig] driver log content : #%s#\n", firstCompilationDriverLog.c_str());
//     EXPECT_TRUE(!containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored") &&
//                 !containsCacheStatus(firstCompilationDriverLog, "cache_status_t::found"));

//     // second compilation
//     OV_ASSERT_NO_THROW(execNet = m_core->compile_model(m_function, target_device, m_configuration));
//     std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("==[2.2][OVCacheConfig] driver log content : #%s#\n", secondCompilationDriverLog.c_str());
//     EXPECT_TRUE(!containsCacheStatus(secondCompilationDriverLog, "cache_status_t::stored") &&
//                 !containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));
// }

// TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig) {
//     ze_graph_dditable_ext_decorator& graph_ddi_table_ext = m_initStruct->getGraphDdiTable();

//     m_configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
//     ov::CompiledModel execNet;
//     m_function = getConstantGraph();

//     // first compilation
//     OV_ASSERT_NO_THROW(execNet = m_core->compile_model(m_function, target_device, m_configuration));
//     std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("==[3.1][bypassConfig] driver log content : #%s#\n", firstCompilationDriverLog.c_str());
//     EXPECT_TRUE(!containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored") &&
//                 !containsCacheStatus(firstCompilationDriverLog, "cache_status_t::found"));

//     // second compilation
//     OV_ASSERT_NO_THROW(execNet = m_core->compile_model(m_function, target_device, m_configuration));
//     std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("==[3.1][bypassConfig] driver log content : #%s#\n", secondCompilationDriverLog.c_str());
//     EXPECT_TRUE(!containsCacheStatus(secondCompilationDriverLog, "cache_status_t::stored") &&
//                 !containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));
// }

}  // namespace behavior
}  // namespace test
}  // namespace ov