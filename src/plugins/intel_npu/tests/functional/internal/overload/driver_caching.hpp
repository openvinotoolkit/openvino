#include <gtest/gtest.h>

#include <common_test_utils/test_assertions.hpp>
#include <sstream>

#include "base/ov_behavior_test_utils.hpp"

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include <openvino/opsets/opset1.hpp>

#include "stdio.h" 
#include <stdlib.h>

#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/config/config.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

#include <filesystem>
#include <chrono>

//check cache folder
#ifdef WIN32
#include "Shlobj.h"
#include "shlobj_core.h"
#include "objbase.h"
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace ov {
namespace test {
namespace behavior {

bool containsCacheStatus(const std::string& str, const std::string cmpstr);   
void checkCacheDirectory(); 

inline std::shared_ptr<ov::Model> getConstantGraph() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::printf("=========print now timestamp #%ld#\n", timestamp);
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add" + std::to_string(timestamp));
    return std::make_shared<ov::Model>(ov::OutputVector{add->output(0)}, ov::ParameterVector{param});
}

bool containsCacheStatus(const std::string& str, const std::string cmpstr) {  
    return str.find(cmpstr) != std::string::npos;  
}

void checkCacheDirectory() {
    std::filesystem::path path{};
#ifdef WIN32
    wchar_t* local = nullptr;
    auto result = SHGetKnownFolderPath( FOLDERID_LocalAppData, 0, NULL, &local );

    if(SUCCEEDED(result)) {
        path = std::filesystem::path( L"\\\\?\\" + std::wstring( local ) + +L"\\Intel\\NPU" );
        CoTaskMemFree( local );
    }
#else
    const char *env = getenv("ZE_INTEL_NPU_CACHE_DIR");
    if (env) {
        path = std::filesystem::path(env);
    } else {
        env = getenv("HOME");
        if (env) {
            path = std::filesystem::path(env) / ".cache/ze_intel_npu_cache";
        } else {
            path = std::filesystem::current_path() / ".cache/ze_intel_npu_cache";
        }
    }
#endif

    std::printf("======check cache path: #%s#\n", path.string().c_str());
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::printf("  ====cache content: #%s# \n", entry.path().string().c_str());
        }
    }
}

typedef std::tuple<std::string,                 // Device name
                   ov::AnyMap                   // Config
                   >
    CompileAndModelCachingParams;

class CompileAndDriverCaching : public testing::WithParamInterface<CompileAndModelCachingParams>,
                                public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndModelCachingParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        initStruct = std::make_shared<::intel_npu::ZeroInitStructsHolder>();
        if (!initStruct) {
            GTEST_SKIP() << "ZeroInitStructsHolder init failed, ZeroInitStructsHolder is a nullptr";
        }

        ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
        uint32_t graphDdiExtVersion = graph_ddi_table_ext.version();
        if (graphDdiExtVersion < ZE_GRAPH_EXT_VERSION_1_5) {
            GTEST_SKIP() << "Skipping test for Driver version less than 1.5, current driver version: " << graphDdiExtVersion;
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

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct;
    std::string m_cachedir;
};

TEST_P(CompileAndDriverCaching, CompilationCacheWithEmptyConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    std::string driverLogInitContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);//this line not used.
    std::printf("==[1.1][EmptyConfig] driver log content : #%s#\n", driverLogInitContent.c_str());
    
    ov::CompiledModel execNet;
    function = getConstantGraph();

    //first compilation.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    //To avoid problems with repeatedly calling functiontest
    std::printf("==[1.2][EmptyConfig] driver log content : #%s#\n", firstCompilationDriverLog.c_str());
    EXPECT_TRUE(containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored"));

    checkCacheDirectory();
    //second compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1.3][EmptyConfig] driver log content : #%s#\n", secondCompilationDriverLog.c_str());
    EXPECT_TRUE(containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));

    std::printf("==[1.4]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    checkCacheDirectory();
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithOVCacheConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    
    //Check the initial state if this testp is called separately
    std::string driverLogInitContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.1][OVCacheConfig] driver log content : #%s#\n", driverLogInitContent.c_str());

    configuration[ov::cache_dir.name()] = "./testCacheDir";
    m_cachedir = configuration[ov::cache_dir.name()].as<std::string>();
    ov::CompiledModel execNet;
    function = getConstantGraph();

    //first compilation will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.2][OVCacheConfig] driver log content : #%s#\n", firstCompilationDriverLog.c_str());
    EXPECT_TRUE(firstCompilationDriverLog == driverLogInitContent);

    checkCacheDirectory();
    //second  compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.3][OVCacheConfig] driver log content : #%s#\n", secondCompilationDriverLog.c_str());
    EXPECT_TRUE(secondCompilationDriverLog == driverLogInitContent);

    std::printf("==[2.4]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    checkCacheDirectory();
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    //Check the initial state if this testp is called separately
    std::string driverLogInitContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.1][bypassConfig] driver log content1 : #%s#\n", driverLogInitContent.c_str());

    configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
    ov::CompiledModel execNet;
    function = getConstantGraph();

    //first compilation will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.2][bypassConfig] driver log content1 : #%s#\n", firstCompilationDriverLog.c_str());
    EXPECT_TRUE(firstCompilationDriverLog == driverLogInitContent);

    checkCacheDirectory();
    //second compilation
    auto startSecond = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.3][bypassConfig] driver log content1 : #%s#\n", secondCompilationDriverLog.c_str());
    EXPECT_TRUE(secondCompilationDriverLog == driverLogInitContent);

    std::printf("==[3.4]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    checkCacheDirectory();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov