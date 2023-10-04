// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <ie_common.h>
#include <thread>

#include "behavior/plugin/caching_tests.hpp"
#include "common_test_utils/file_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/subgraph_builders.hpp"

using namespace InferenceEngine::details;
using namespace InferenceEngine;
using namespace ::testing;
using namespace std::placeholders;

namespace LayerTestsDefinitions {

static std::shared_ptr<ngraph::Function> simple_function_multiply(ngraph::element::Type type, size_t batchSize) {
    // Create Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset6::Parameter>(type, ngraph::Shape{batchSize, 2});
    data->set_friendly_name("Parameter");

    auto constant = ngraph::opset6::Constant::create(type, ngraph::Shape{1}, {2});
    constant->set_friendly_name("constant");
    auto mul = std::make_shared<ngraph::opset6::Multiply>(data, constant);
    mul->set_friendly_name("mul");

    // Create Result operation
    auto res = std::make_shared<ngraph::opset6::Result>(mul);
    res->set_friendly_name("res");

    // Create nGraph function
    auto func = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
    func->set_friendly_name("function");
    return func;
}

static std::shared_ptr<ngraph::Function> simple_function_relu(ngraph::element::Type type, size_t batchSize) {
    // Create Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset6::Parameter>(type, ngraph::Shape{batchSize, 2});
    data->set_friendly_name("Parameter");

    auto relu = std::make_shared<ngraph::opset6::Relu>(data);
    relu->set_friendly_name("relu");

    // Create Result operation
    auto res = std::make_shared<ngraph::opset6::Result>(relu);
    res->set_friendly_name("res");

    // Create nGraph function
    auto func = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
    func->set_friendly_name("function");
    return func;
}

ngraphFunctionGenerator LoadNetworkCacheTestBase::inputShapeWrapper(ngraphFunctionIS fun, std::vector<size_t> inputShape) {
    return [fun, inputShape](ngraph::element::Type type, std::size_t batchSize) {
        auto shape = inputShape;
        shape[0] = batchSize;
        return fun(shape, type);
    };
}

std::vector<nGraphFunctionWithName> LoadNetworkCacheTestBase::getNumericTypeOnlyFunctions() {
    std::vector<nGraphFunctionWithName> res;
    res.push_back(nGraphFunctionWithName { simple_function_multiply, "SimpleFunctionMultiply"});
    res.push_back(nGraphFunctionWithName { simple_function_relu, "SimpleFunctionRelu"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeConvPoolRelu, {1, 1, 32, 32}),
        "ConvPoolRelu"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcat, {1, 4, 20, 20}),
        "SplitConvConcat"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeKSOFunction, {1, 4, 20, 20}),
        "KSOFunction"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSingleConv, {1, 3, 24, 24}),
        "SingleConv"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::make2InputSubtract, {1, 3, 24, 24}),
        "2InputSubtract"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeNestedSplitConvConcat, {1, 4, 20, 20}),
        "NestedSplitConvConcat"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatInputInBranch, {1, 4, 20, 20}),
        "SplitConvConcatInputInBranch"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatNestedInBranch, {1, 4, 20, 20}),
        "SplitConvConcatNestedInBranch"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatNestedInBranchNestedOut, {1, 4, 20, 20}),
        "SplitConvConcatNestedInBranchNestedOut"});
    res.push_back(nGraphFunctionWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeConvBias, {1, 3, 24, 24}),
        "ConvBias"});
    res.push_back(nGraphFunctionWithName{
        inputShapeWrapper(ngraph::builder::subgraph::makeMatMulBias, {1, 3, 24, 24}),
        "MatMulBias" });
    return res;
}

std::vector<nGraphFunctionWithName> LoadNetworkCacheTestBase::getAnyTypeOnlyFunctions() {
    std::vector<nGraphFunctionWithName> res;

    return res;
}

std::vector<nGraphFunctionWithName> LoadNetworkCacheTestBase::getFloatingPointOnlyFunctions() {
    std::vector<nGraphFunctionWithName> res;
    res.push_back(nGraphFunctionWithName { [](ngraph::element::Type type, size_t batchSize) {
        return ngraph::builder::subgraph::makeTIwithLSTMcell(type, batchSize);
    }, "TIwithLSTMcell1"});
    return res;
}

std::vector<nGraphFunctionWithName> LoadNetworkCacheTestBase::getNumericAnyTypeFunctions() {
    std::vector<nGraphFunctionWithName> funcs = LoadNetworkCacheTestBase::getAnyTypeOnlyFunctions();
    std::vector<nGraphFunctionWithName> numericType = LoadNetworkCacheTestBase::getNumericTypeOnlyFunctions();
    funcs.insert(funcs.end(), numericType.begin(), numericType.end());

    return funcs;
}

std::vector<nGraphFunctionWithName> LoadNetworkCacheTestBase::getStandardFunctions() {
    std::vector<nGraphFunctionWithName> funcs = LoadNetworkCacheTestBase::getAnyTypeOnlyFunctions();
    std::vector<nGraphFunctionWithName> numericType = LoadNetworkCacheTestBase::getNumericTypeOnlyFunctions();
    funcs.insert(funcs.end(), numericType.begin(), numericType.end());
    std::vector<nGraphFunctionWithName> floatType = LoadNetworkCacheTestBase::getFloatingPointOnlyFunctions();
    funcs.insert(funcs.end(), floatType.begin(), floatType.end());

    return funcs;
}

bool LoadNetworkCacheTestBase::importExportSupported(InferenceEngine::Core& ie) const {
    auto supportedMetricKeys = ie.GetMetric(targetDevice, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
    auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(),
                        METRIC_KEY(IMPORT_EXPORT_SUPPORT));
    auto supported = (it != supportedMetricKeys.end()) &&
                     ie.GetMetric(targetDevice, METRIC_KEY(IMPORT_EXPORT_SUPPORT)).as<bool>();
    return supported;
}

std::string LoadNetworkCacheTestBase::getTestCaseName(testing::TestParamInfo<loadNetworkCacheParams> obj) {
    auto param = obj.param;
    auto funcName = std::get<1>(std::get<0>(param));
    auto precision = std::get<1>(param);
    auto batchSize = std::get<2>(param);
    auto deviceName = std::get<3>(param);
    std::replace(deviceName.begin(), deviceName.end(), ':', '.');
    return funcName + "_" + ngraph::element::Type(precision).get_type_name() + "_batch" + std::to_string(batchSize) + "_" + deviceName;
}

void LoadNetworkCacheTestBase::SetUp() {
    nGraphFunctionWithName funcPair;
    std::tie(funcPair, m_precision, m_batchSize, targetDevice) = GetParam();
    target_device  = targetDevice;
    APIBaseTest::SetUp();
    auto fGen = std::get<0>(funcPair);
    m_functionName = std::get<1>(funcPair);
    function = fGen(m_precision, m_batchSize);

    std::stringstream ss;
    auto hash = std::hash<std::string>()(LayerTestsUtils::LayerTestsCommon::GetTestName());
    ss << "testCache_" << std::to_string(hash) << "_" << std::this_thread::get_id() << "_" << GetTimestamp();
    for (auto& iter : configuration) {
        ss << "_" << iter.first << "_" << iter.second << "_";
    }
    m_cacheFolderName = ss.str();
    core->SetConfig({{CONFIG_KEY(CACHE_DIR), {}}});
}

void LoadNetworkCacheTestBase::TearDown() {
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "blob");
    ov::test::utils::removeDir(m_cacheFolderName);
    core->SetConfig({{CONFIG_KEY(CACHE_DIR), {}}});
    APIBaseTest::TearDown();
}

void LoadNetworkCacheTestBase::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto compareOutputs = [&](const std::vector<InferenceEngine::Blob::Ptr>& expected,
                              const std::vector<InferenceEngine::Blob::Ptr>& actual) {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < expected.size(); i++) {
            const auto& expPtr = expected[i];
            const auto& actPtr = actual[i];
            ASSERT_NO_THROW(Compare(expPtr, actPtr));
        }
    };
    if (!function) {
        GTEST_FAIL() << "Can't create function " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
    }
    if (!importExportSupported(*core)) {
        GTEST_FAIL() << "Plugin doesn't support import and export - skipping test" << std::endl;
    }
    cnnNetwork = CNNNetwork{function};
    ConfigureNetwork();
    try {
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
        GenerateInputs();
        Infer();
    } catch (const Exception &ex) {
        GTEST_FAIL() << "Can't loadNetwork without cache for " << m_functionName << " with precision " << m_precision.get_type_name()  << "\n"
            << "Exception [" << ex.what() << "]" << std::endl;
    } catch (...) {
        GTEST_FAIL() << "Can't loadNetwork without cache for " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
    }
    auto originalOutputs = GetOutputs();

    for (int i = 0; i < 2; i++) {
        // Step 2: Load with cache. Export or import shall not throw
        executableNetwork = {}; // Destroy network object
        inferRequest = {};
        {
            core->SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheFolderName}});
            ASSERT_NO_THROW(executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration));
            GenerateInputs();
            ASSERT_NO_THROW(Infer());
        }
        // cache is created and reused
        ASSERT_EQ(ov::test::utils::listFilesWithExt(m_cacheFolderName, "blob").size(), 1);
        compareOutputs(originalOutputs, GetOutputs());
    }
}

TEST_P(LoadNetworkCacheTestBase, CompareWithRefImpl) {
    Run();
}

std::string LoadNetworkCompiledKernelsCacheTest::getTestCaseName(testing::TestParamInfo<compileKernelsCacheParams> obj) {
    auto param = obj.param;
    std::string deviceName;
    std::pair<std::map<std::string, std::string>, std::string> userConfig;
    std::tie(deviceName, userConfig) = obj.param;
    std::replace(deviceName.begin(), deviceName.end(), ':', '.');
    std::map<std::string, std::string> confstr = userConfig.first;
    std::ostringstream result;
    result << "device_name=" << deviceName << "_";
    for (auto& tmp : confstr) {
        result << tmp.first << "_" << tmp.second << "_";
    }
    result << userConfig.second;
    return result.str();
}

TEST_P(LoadNetworkCompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinaries) {
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    ie->SetConfig({{ CONFIG_KEY(CACHE_DIR), cache_path }});
    try {
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
        execNet = {};
        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(ov::test::utils::directoryExists(cache_path)) << "Directory with cached kernels doesn't exist";
        for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            ASSERT_GT(ov::test::utils::removeFilesWithExt(cache_path, ext), 0);
        }
        // Remove directory and check that it doesn't exist anymore
        ASSERT_EQ(ov::test::utils::removeDir(cache_path), 0);
        ASSERT_FALSE(ov::test::utils::directoryExists(cache_path));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (ov::test::utils::directoryExists(cache_path)) {
            for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            ASSERT_GE(ov::test::utils::removeFilesWithExt(cache_path, ext), 0);
            }
            ASSERT_EQ(ov::test::utils::removeDir(cache_path), 0);
        }
        FAIL() << ex.what() << std::endl;
    }
}

TEST_P(LoadNetworkCompiledKernelsCacheTest, TwoNetworksWithSameModelCreatesSameCache) {
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    // Create two CNNNetwork from same ngraph::Function
    InferenceEngine::CNNNetwork cnnNet1(function);
    InferenceEngine::CNNNetwork cnnNet2(function);
    ie->SetConfig({{ CONFIG_KEY(CACHE_DIR), cache_path }});
    try {
        // Load 1st CNNNetwork
        auto execNet1 = ie->LoadNetwork(cnnNet1, targetDevice, configuration);
        size_t n_cache_files = 0;
        execNet1 = {};
        for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            n_cache_files += ov::test::utils::listFilesWithExt(cache_path, ext).size();
        }

        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(ov::test::utils::directoryExists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Load 2nd CNNNetwork
        auto execNet2 = ie->LoadNetwork(cnnNet2, targetDevice, configuration);
        execNet2 = {};
        size_t n_cache_files_compare = 0;
        // Check that two loaded networks with same function creates same caches
        for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            n_cache_files_compare += ov::test::utils::listFilesWithExt(cache_path, ext).size();
            ASSERT_TRUE(ov::test::utils::removeFilesWithExt(cache_path, ext));
        }

        ASSERT_EQ(n_cache_files_compare, n_cache_files);

        // Remove directory and check that it doesn't exist anymore
        ASSERT_EQ(ov::test::utils::removeDir(cache_path), 0);
        ASSERT_FALSE(ov::test::utils::directoryExists(cache_path));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (ov::test::utils::directoryExists(cache_path)) {
            for (auto& ext : m_extList) {
                // Check that folder contains cache files and remove them
                ASSERT_GE(ov::test::utils::removeFilesWithExt(cache_path, ext), 0);
            }
            ASSERT_EQ(ov::test::utils::removeDir(cache_path), 0);
        }
        FAIL() << ex.what() << std::endl;
    }
}


#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
TEST_P(LoadNetworkCompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinariesUnicodePath) {
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix  = L"_" + ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring cache_path_w = ov::test::utils::stringToWString(cache_path) + postfix;

        try {
            auto cache_path_mb = ov::util::wstring_to_string(cache_path_w);
            ie->SetConfig({{ CONFIG_KEY(CACHE_DIR), cache_path_mb }});
            // Load CNNNetwork to target plugins
            auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
            execNet = {};
            // Check that directory with cached kernels exists after loading network
            ASSERT_TRUE(ov::test::utils::directoryExists(cache_path_w)) << "Directory with cached kernels doesn't exist";
            // Check that folder contains cache files and remove them
            for (auto& ext : m_extList) {
                // Check that folder contains cache files and remove them
                ASSERT_GT(ov::test::utils::removeFilesWithExt(cache_path_w, ov::test::utils::stringToWString(ext)), 0);
            }
            //ASSERT_GT(ov::test::utils::removeFilesWithExt(cache_path_w, L"cl_cache"), 0);
            // Remove directory and check that it doesn't exist anymore
            ASSERT_EQ(ov::test::utils::removeDir(cache_path_w), 0);
            ASSERT_FALSE(ov::test::utils::directoryExists(cache_path_w));
        } catch (std::exception& ex) {
            // Cleanup in case of any exception
            if (ov::test::utils::directoryExists(cache_path_w)) {
                for (auto& ext : m_extList) {
                    // Check that folder contains cache files and remove them
                    ASSERT_GE(ov::test::utils::removeFilesWithExt(cache_path_w, ov::test::utils::stringToWString(ext)), 0);
                }
                ASSERT_EQ(ov::test::utils::removeDir(cache_path_w), 0);
            }
            FAIL() << ex.what() << std::endl;
        }
    }
}
#endif
} // namespace LayerTestsDefinitions
