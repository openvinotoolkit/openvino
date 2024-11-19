#include <gtest/gtest.h>

#include <chrono>
#include <common_test_utils/test_assertions.hpp>
#include <openvino/opsets/opset1.hpp>

#include "base/ov_behavior_test_utils.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

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

typedef std::tuple<std::string,  // Device name
                   ov::AnyMap    // Config
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

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct;
    std::string m_cachedir;
};

TEST_P(CompileAndDriverCaching, CompilationCacheWithEmptyConfig) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    ov::CompiledModel execNet;
    function = getConstantGraph();

    // first compilation
    auto startFirstCompilationTime = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    ;
    EXPECT_TRUE(containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored"));
    auto endFirstCompilationTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirstCompilation = endFirstCompilationTime - startFirstCompilationTime;

    // second compilation
    auto startSecondCompilationTime = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecondCompilationTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecondCompilation = startSecondCompilationTime - endSecondCompilationTime;
    std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    EXPECT_TRUE(containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));

    EXPECT_LT(durationSecondCompilation, durationFirstCompilation)
        << "The duration of the second compilation should be less than the first due to UMD Caching.";
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithOVCacheConfig) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    configuration[ov::cache_dir.name()] = "./testCacheDir";
    m_cachedir = configuration[ov::cache_dir.name()].as<std::string>();
    ov::CompiledModel execNet;
    function = getConstantGraph();

    // first compilation
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    EXPECT_TRUE(!containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored") &&
                !containsCacheStatus(firstCompilationDriverLog, "cache_status_t::found"));

    // second compilation
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    EXPECT_TRUE(!containsCacheStatus(secondCompilationDriverLog, "cache_status_t::stored") &&
                !containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
    ov::CompiledModel execNet;
    function = getConstantGraph();

    // first compilation
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    std::string firstCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    EXPECT_TRUE(!containsCacheStatus(firstCompilationDriverLog, "cache_status_t::stored") &&
                !containsCacheStatus(firstCompilationDriverLog, "cache_status_t::found"));

    // second compilation
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    std::string secondCompilationDriverLog = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    EXPECT_TRUE(!containsCacheStatus(secondCompilationDriverLog, "cache_status_t::stored") &&
                !containsCacheStatus(secondCompilationDriverLog, "cache_status_t::found"));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov