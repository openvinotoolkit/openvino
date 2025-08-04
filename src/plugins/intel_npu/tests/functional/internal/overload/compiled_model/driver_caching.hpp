#include <gtest/gtest.h>

#include <common_test_utils/test_assertions.hpp>
#include <sstream>

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include <openvino/opsets/opset1.hpp>
#include "openvino/core/log_util.hpp"

#include "stdio.h" 
#include <stdlib.h>

#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

#include "intel_npu/config/config.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

#include <filesystem>
#include <chrono>

#include "overload/overload_test_utils_npu.hpp"

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ov::Model> getConstantGraph() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add" + std::to_string(timestamp));
    return std::make_shared<ov::Model>(ov::OutputVector{add->output(0)}, ov::ParameterVector{param});
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
                result << "_";
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
        core->set_property(ov::cache_dir());

        ov::test::utils::PluginCache::get().reset();
        APIBaseTest::TearDown();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct;
    std::string m_cachedir;
    std::string m_cacheFolderName;
};

TEST_P(CompileAndDriverCaching, CompilationCacheWithEmptyConfig) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    std::string driverLogInitContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1.1][EmptyConfig] driver log content : #%s#\n", driverLogInitContent.c_str());
    
    ov::CompiledModel execNet;
    function = getConstantGraph();

    //First compilation
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    //To avoid problems with repeatedly calling functiontest
    std::printf("==[1.2][EmptyConfig] driver log content : #%s#\n", firstCompilationDriverLog.c_str());
    // EXPECT_TRUE(containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored"));

    //Second compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1.3][EmptyConfig] driver log content : #%s#\n", secondCompilationDriverLog.c_str());
    // EXPECT_TRUE(containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));

    std::printf("==[1.4]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    EXPECT_LT(durationSecond, durationFirst)
        << "The duration of the second compilation should be less than the first due to UMD Caching.";
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithOVCacheConfig) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    
    //Check the initial state if this testp is called separately
    std::string driverLogInitContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.1][OVCacheConfig] driver log content : #%s#\n", driverLogInitContent.c_str());

    // configuration[ov::cache_dir.name()] = "./testCacheDir";
    m_cachedir = configuration[ov::cache_dir.name()].as<std::string>();
    ov::CompiledModel execNet;
    function = getConstantGraph();

    //First compilation should take a while and it should fill the model cache
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.2][OVCacheConfig] driver log content : #%s#\n", firstCompilationDriverLog.c_str());
    // EXPECT_TRUE(containsCacheStatus(firstCompilationDriverLog, driverLogInitContent));

    //Second compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.3][OVCacheConfig] driver log content : #%s#\n", secondCompilationDriverLog.c_str());
    // EXPECT_TRUE(containsCacheStatus(secondCompilationDriverLog, driverLogInitContent));

    std::printf("==[2.4]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig) {
    std::cout << "==[3.1][bypassConfig] set bypass_umd_caching: " << "\n";
    ov::CompiledModel execNet1;
    ov::CompiledModel execNet2;
    ov::CompiledModel execNet3;
    
    function = getConstantGraph();

    //First compilation
    auto start = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet1 = core->compile_model(function, target_device, configuration));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = end - start;
    std::printf("==[3.2][bypassConfig] First compilation done\n");


    execNet1={}; //is bug?

    //Second compilation
    start = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet1 = core->compile_model(function, target_device, configuration));
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = end - start;
    std::printf("==[3.3][bypassConfig] Second compilation done\n");

    execNet1={};

    //Third compilation
    start = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet1 = core->compile_model(function, target_device, configuration));
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationThird = end - start;
    std::printf("==[3.4][bypassConfig] Third compilation done\n");

    std::printf("==[3.5]testsuit time (1): %f, (2): %f, (3): %f\n", durationFirst.count(), durationSecond.count(), durationThird.count());
}


TEST_P(CompileAndDriverCaching, StandardOutputRedirect) {
    std::stringstream buffer;

    std::cout << "== Start compile\n";
    std::streambuf* oldOutput = std::cout.rdbuf(buffer.rdbuf()); // Redirect cout

    ov::CompiledModel execNet;
    function = getConstantGraph();
    
    //First compilation
    auto start = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto end = std::chrono::high_resolution_clock::now();

    std::cout.rdbuf(oldOutput); // Restore original buffer
    std::cout << "== End compile\n";

    std::string capturedOutput = buffer.str();
    std::cout << "!!!=============!!! Captured log: \n";
    std::cout << capturedOutput;
    std::cout << "!!!=============!!! Captured log end\n";
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
