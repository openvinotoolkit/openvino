#include <gtest/gtest.h>

#include <common_test_utils/test_assertions.hpp>
#include <sstream>

// #include "shared_test_classes/base/ov_subgraph.hpp"s
#include "base/ov_behavior_test_utils.hpp"

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/properties.hpp"

#include "intel_npu/utils/zero/zero_utils.hpp"

#include "stdio.h" //
#include <stdlib.h>// env setting

#include "zero_backend.hpp"
#include "zero_types.hpp"

// #include "intel_npu/config/common.hpp"
// #include "intel_npu/config/compiler.hpp"
// #include "intel_npu/config/runtime.hpp"
// #include "intel_npu/config/config.hpp"

// #include "/home/dl5w050/vpux/openvino/src/plugins/intel_npu/src/backend/include/zero_backend.hpp"
// #include "/home/dl5w050/vpux/openvino/src/plugins/intel_npu/src/al/include/intel_npu/config/config.hpp"

#include <filesystem>


#include <chrono> // cal time

namespace ov {
namespace test {
namespace behavior {

typedef std::tuple<std::shared_ptr<ov::Model>,  // Model
                   std::string,                 // Device name
                   ov::AnyMap                   // Config
                   >
    CompileAndModelCachingParams;

inline std::shared_ptr<ov::Model> getConstantGraph() {
    ResultVector results;
    ParameterVector params;
    auto op = std::make_shared<ov::op::v1::Add>(opset8::Constant::create(ov::element::f32, {1}, {1}),
                                                opset8::Constant::create(ov::element::f32, {1}, {1}));
    op->set_friendly_name("Add");
    auto res = std::make_shared<ov::op::v0::Result>(op);
    res->set_friendly_name("Result");
    res->get_output_tensor(0).set_names({"tensor_output"});
    results.push_back(res);
    return std::make_shared<Model>(results, params);
}

std::string generateCacheDirName(const std::string& test_name) {
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto hash = std::to_string(std::hash<std::string>()(test_name));
    std::stringstream ss;
    auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
    ss << hash << "_"
        << "_" << ts.count();
    return ss.str();
}

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

bool containsCacheStatus(const std::string& str) {  
    return str.find("cache_status_t::stored") != std::string::npos;  
}



inline std::vector<std::string> listFilesWithExt(const std::string& path) {
    struct dirent* ent;
    DIR* dir = opendir(path.c_str());
    std::vector<std::string> res;
    if (dir != nullptr) {
        while ((ent = readdir(dir)) != NULL) {
            auto file = ov::test::utils::makePath(path, std::string(ent->d_name));
            struct stat stat_path;
            stat(file.c_str(), &stat_path);
            //cache not contian file extension.
            if (!S_ISDIR(stat_path.st_mode) && ov::test::utils::endsWith(file, "")) {
                res.push_back(std::move(file));
            }
        }
        closedir(dir);
    }
    return res;
}

class CompileAndDriverCaching : public testing::WithParamInterface<CompileAndDriverCaching>,
                                public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndModelCachingParams> obj) {
        std::shared_ptr<ov::Model> model;
        std::string targetDevice;
        ov::AnyMap configuration; //using AnyMap = std::map<std::string, Any>;
        //const std::map<std::string, std::string>& rawConfig
        //_globalConfig.update(rawConfig);
        std::tie(model, targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            // using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        std::printf("<====local test name> %s\n", result.str().c_str());
        return result.str();
    }

    void SetUp() override {
        std::tie(function, target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        //ov::AnyMap configuration; //using AnyMap = std::map<std::string, Any>;
        //const std::map<std::string, std::string>& rawConfig
        //_globalConfig.update(rawConfig);
        
        // const std::map<std::string, std::string> stringConfig = any_copy(configuration);
        // // Config config;
        // config.update(stringConfig);
        // std::shared_ptr<ZeroEngineBackend> zeroBackend = nullptr;
        // zeroBackend = std::make_shared<intel_npu::ZeroEngineBackend>(config);
        zeroBackend = std::make_shared<intel_npu::ZeroEngineBackend>();
        if (!zeroBackend) {
            GTEST_SKIP() << "LevelZeroCompilerAdapter init failed to cast zeroBackend, zeroBackend is a nullptr";
        }

        graph_ddi_table_ext = zeroBackend->getGraphDdiTable();

        APIBaseTest::SetUp();
    }

    void TearDown() override {
        if (!m_cache_dir.empty() && !std::filesystem::exists(m_cache_dir)) {
            std::filesystem::remove_all(m_cache_dir);
            //ov::test::utils::removeDir(m_cache_dir);
        }

        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }

private:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;

    // intel_npu::Config& config;
    std::shared_ptr<intel_npu::ZeroEngineBackend> zeroBackend;
    ze_graph_dditable_ext_curr_t& graph_ddi_table_ext;

    std::string m_cache_dir; //it is need to be distinguished on Windows and Linux?
};

TEST_P(CompileAndDriverCaching, CompilationCacheFlag) {
    //TODO: check driver version, if less than 1.5 will not support cache feature.
    
    uint32_t graphDdiExtVersion = graph_ddi_table_ext.version();
    if (graphDdiExtVersion < ZE_GRAPH_EXT_VERSION_1_5) {
        GTEST_SKIP() << "Skipping test for Driver version less than 1.5, current driver version: " << graphDdiExtVersion;
    }

    std::string driverLogContent = intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    if ( driverLogContent.find( "result_t::stored" ) != std::string::npos ) {
        std::printf("printf testsuit contain stored");
    }
    EXPECT_TRUE(containsCacheStatus(driverLogContent));
}

#ifdef WIN32
TEST_P(CompileAndDriverCaching, CompilationTwiceOnWindwos) {
    //windows cache dir located on C:\Users\account\AppData\Local\Intel\NPU
    // attempt to create root folder in AppData\Local
    std::filesystem::path path{};
    wchar_t* local = nullptr;
    auto result = SHGetKnownFolderPath( FOLDERID_LocalAppData, 0, NULL, &local );

    if( SUCCEEDED( result ) )
    {
        // prepend to enable long path name support
        path = std::filesystem::path( L"\\\\?\\" + std::wstring( local ) + +L"\\Intel\\NPU" );

        CoTaskMemFree( local );

        if( !std::filesystem::exists(path) )
        {
            std::filesystem::create_directories(path);
        } else {
            std::filesystem::remove_all(path);
        }
    }
    size_t blobCountInitial = -1;
    blobCountInitial = listFilesWithExt(path).size();
    size_t blobCountAfterwards = -1;
    ASSERT_GT(blobCountInitial, 0);
    
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    blobCountAfterwards = listFilesWithExt(path).size();
    if (config.get<CACHE_DIR>().empty() || config.get<BYPASS_UMD_CACHING>()) {
        ASSERT_GT(blobCountInitial, 0);
    } else {
        ASSERT_EQ(blobCountInitial, blobCountAfterwards - 1);
    }

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    double epsilon = 20.0;
    if (config.get<CACHE_DIR>().empty() || config.get<BYPASS_UMD_CACHING>()) {
        EXPECT_NEAR(durationFirst, durationSecond, epsilon);
    } else {
        EXPECT_NEAR(durationFirst, durationSecond, durationFirst / 2.0);
    }

    std::filesystem::remove_all(path);
}

#else

TEST_P(CompileAndDriverCaching, CompilationTwiceOnLinux) {
    //ON linux, cache dir can be set by env variables.
    m_cache_dir = generateCacheDirName(GetTestName());
    auto temp = std::setenv("ZE_INTEL_NPU_CACHE_DIR", m_cache_dir, 1);
    size_t blobCountInitial = -1;
    blobCountInitial = listFilesWithExt(m_cache_dir).size();
    size_t blobCountAfterwards = -1;
    ASSERT_GT(blobCountInitial, 0);
    
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    blobCountAfterwards = listFilesWithExt(m_cache_dir).size();
    if (config.get<CACHE_DIR>().empty() || config.get<BYPASS_UMD_CACHING>()) {
        ASSERT_GT(blobCountInitial, 0);
    } else {
        ASSERT_EQ(blobCountInitial, blobCountAfterwards - 1);
    }

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    double epsilon = 20.0;
    if (config.get<CACHE_DIR>().empty() || config.get<BYPASS_UMD_CACHING>()) {
        EXPECT_NEAR(durationFirst, durationSecond, epsilon);
    } else {
        EXPECT_NEAR(durationFirst, durationSecond, durationFirst / 2.0);
    }
}
#endif


}  // namespace behavior
}  // namespace test
}  // namespace ov
