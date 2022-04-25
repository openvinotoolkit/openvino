// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <thread>

#include "behavior/ov_plugin/caching_tests.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

#define GTEST_COUT std::cout << "[          ] [ INFO ] "

namespace ov {
namespace test {
namespace behavior {

static std::shared_ptr<ov::Model> simple_function_multiply(ov::element::Type type, size_t batchSize) {
    // Create Parameter operation with static shape
    auto data = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{batchSize, 2});
    data->set_friendly_name("Parameter");

    auto constant = ov::op::v0::Constant::create(type, ov::Shape{1}, {2});
    constant->set_friendly_name("constant");
    auto mul = std::make_shared<ov::op::v1::Multiply>(data, constant);
    mul->set_friendly_name("mul");

    // Create Result operation
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    res->set_friendly_name("res");

    // Create nGraph function
    auto func = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
    func->set_friendly_name("function");
    return func;
}

static std::shared_ptr<ov::Model> simple_function_relu(ov::element::Type type, size_t batchSize) {
    // Create Parameter operation with static shape
    auto data = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{batchSize, 2});
    data->set_friendly_name("Parameter");

    auto relu = std::make_shared<ov::op::v0::Relu>(data);
    relu->set_friendly_name("relu");

    // Create Result operation
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    res->set_friendly_name("res");

    // Create nGraph function
    auto func = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data});
    func->set_friendly_name("function");
    return func;
}

std::vector<ovModelWithName> CompileModelCacheTestBase::getStandardFunctions() {
    // Wrapper of most part of available builder functions
    using ovModelIS = std::function<std::shared_ptr<ov::Model>(std::vector<size_t> inputShape,
                                                                      ov::element::Type_t type)>;
    auto inputShapeWrapper = [](ovModelIS fun, std::vector<size_t> inputShape) {
        return [fun, inputShape](ngraph::element::Type type, std::size_t batchSize) {
            auto shape = inputShape;
            shape[0] = batchSize;
            return fun(shape, type);
        };
    };

    std::vector<ovModelWithName> res;
    res.push_back(ovModelWithName { simple_function_multiply, "SimpleFunctionMultiply"});
    res.push_back(ovModelWithName { simple_function_relu, "SimpleFunctionRelu"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeConvPoolRelu, {1, 1, 32, 32}),
        "ConvPoolRelu"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcat, {1, 4, 20, 20}),
        "SplitConvConcat"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeKSOFunction, {1, 4, 20, 20}),
        "KSOFunction"});
    res.push_back(ovModelWithName { [](ngraph::element::Type type, size_t batchSize) {
        return ngraph::builder::subgraph::makeTIwithLSTMcell(type, batchSize);
    }, "TIwithLSTMcell1"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSingleConv, {1, 3, 24, 24}),
        "SingleConv"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::make2InputSubtract, {1, 3, 24, 24}),
        "2InputSubtract"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeNestedSplitConvConcat, {1, 4, 20, 20}),
        "NestedSplitConvConcat"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatInputInBranch, {1, 4, 20, 20}),
        "SplitConvConcatInputInBranch"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatNestedInBranch, {1, 4, 20, 20}),
        "SplitConvConcatNestedInBranch"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatNestedInBranchNestedOut, {1, 4, 20, 20}),
        "SplitConvConcatNestedInBranchNestedOut"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeConvBias, {1, 3, 24, 24}),
        "ConvBias"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeReadConcatSplitAssign, {1, 1, 2, 4}),
        "ReadConcatSplitAssign"});
    res.push_back(ovModelWithName{
        inputShapeWrapper(ngraph::builder::subgraph::makeMatMulBias, {1, 3, 24, 24}),
        "MatMulBias" });

    return res;
}

bool CompileModelCacheTestBase::importExportSupported(ov::Core& core) const {
    auto supportedProperties = core.get_property(targetDevice, ov::supported_properties);
    if (std::find(supportedProperties.begin(), supportedProperties.end(), ov::device::capabilities) == supportedProperties.end()) {
        return false;
    }
    auto device_capabilities = core.get_property(targetDevice, ov::device::capabilities);
    if (std::find(device_capabilities.begin(), device_capabilities.end(), std::string(ov::device::capability::EXPORT_IMPORT)) == device_capabilities.end()) {
        return false;
    }
    return true;
}

std::string CompileModelCacheTestBase::getTestCaseName(testing::TestParamInfo<compileModelCacheParams> obj) {
    auto param = obj.param;
    auto funcName = std::get<1>(std::get<0>(param));
    auto precision = std::get<1>(param);
    auto batchSize = std::get<2>(param);
    auto deviceName = std::get<3>(param);
    return funcName + "_" + ngraph::element::Type(precision).get_type_name() + "_batch" + std::to_string(batchSize) + "_" + deviceName;
}

void CompileModelCacheTestBase::SetUp() {
    ovModelWithName funcPair;
    std::tie(funcPair, m_precision, m_batchSize, targetDevice, configuration) = GetParam();
    auto fGen = std::get<0>(funcPair);
    m_functionName = std::get<1>(funcPair);
    try {
        function = fGen(m_precision, m_batchSize);
    } catch (...) {
        GTEST_SKIP();
    }

    std::stringstream ss;
    auto hash = std::hash<std::string>()(GetTestName());
    ss << "testCache_" << std::to_string(hash) << "_" << std::this_thread::get_id() << "_" << GetTimestamp();
    for (auto& iter : configuration) {
        ss << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
    }
    m_cacheFolderName = ss.str();
    core->set_property(ov::cache_dir());
}

void CompileModelCacheTestBase::TearDown() {
    CommonTestUtils::removeFilesWithExt(m_cacheFolderName, "blob");
    std::remove(m_cacheFolderName.c_str());
    core->set_property(ov::cache_dir());
}

void CompileModelCacheTestBase::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    if (!function) {
        GTEST_COUT << "Can't create function " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
        GTEST_SKIP();
    } else {
        std::vector<ov::Shape> inShapes;
        for (const auto& param : function->get_parameters()) {
            inShapes.push_back(param->get_shape());
        }
        init_input_shapes(static_shapes_to_test_representation(inShapes));
    }
    if ((targetDevice.find("AUTO") == std::string::npos) && !importExportSupported(*core)) {
        GTEST_COUT << "Plugin doesn't support import and export - skipping test" << std::endl;
        GTEST_SKIP();
    }
    configure_model();
    try {
        compiledModel = core->compile_model(function, targetDevice, configuration);
        generate_inputs(targetStaticShapes.front());
        infer();
    } catch (const Exception &ex) {
        GTEST_COUT << "Can't loadNetwork without cache for " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
        GTEST_COUT << "Exception [" << ex.what() << "]" << std::endl;
        GTEST_SKIP();
    } catch (...) {
        GTEST_COUT << "Can't compile network without cache for " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
        GTEST_SKIP(); // skip caching test if such network is not supported by device at all
    }
    auto originalOutputs = get_plugin_outputs();

    for (int i = 0; i < 2; i++) {
        // Step 2: Load with cache. Export or import shall not throw
        compiledModel = {}; // Destroy network object
        {
            core->set_property(ov::cache_dir(m_cacheFolderName));
            ASSERT_NO_THROW(compiledModel = core->compile_model(function, targetDevice, configuration));
            generate_inputs(targetStaticShapes.front());
            ASSERT_NO_THROW(infer());
        }
        // cache is created and reused
        ASSERT_EQ(CommonTestUtils::listFilesWithExt(m_cacheFolderName, "blob").size(), 1);
        compare(originalOutputs, get_plugin_outputs());
    }
}

TEST_P(CompileModelCacheTestBase, CompareWithRefImpl) {
    run();
}

std::string CompiledKernelsCacheTest::getTestCaseName(testing::TestParamInfo<compileKernelsCacheParams> obj) {
    auto param = obj.param;
    std::string deviceName;
    std::pair<ov::AnyMap, std::string> userConfig;
    std::tie(deviceName, userConfig) = obj.param;
    auto properties = userConfig.first;
    std::ostringstream result;
    result << "device_name=" << deviceName << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    result << userConfig.second;
    return result.str();
}

TEST_P(CompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinaries) {
    core->set_property(ov::cache_dir(cache_path));
    try {
        // Load CNNNetwork to target plugins
        auto execNet = core->compile_model(function, targetDevice, configuration);
        execNet = {};
        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(CommonTestUtils::directoryExists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Check that folder contains cache files and remove them
        for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            ASSERT_GT(CommonTestUtils::removeFilesWithExt(cache_path, ext), 0);
        }
        // Remove directory and check that it doesn't exist anymore
        ASSERT_EQ(CommonTestUtils::removeDir(cache_path), 0);
        ASSERT_FALSE(CommonTestUtils::directoryExists(cache_path));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (CommonTestUtils::directoryExists(cache_path)) {
            for (auto& ext : m_extList) {
                // Check that folder contains cache files and remove them
                ASSERT_GT(CommonTestUtils::removeFilesWithExt(cache_path, ext), 0);
        }
            ASSERT_EQ(CommonTestUtils::removeDir(cache_path), 0);
        }
        FAIL() << ex.what() << std::endl;
    }
}

TEST_P(CompiledKernelsCacheTest, TwoNetworksWithSameModelCreatesSameCache) {
    core->set_property(ov::cache_dir(cache_path));
    try {
        // Load 1st CNNNetwork
        auto execNet1 = core->compile_model(function, targetDevice, configuration);
        execNet1 = {};
        size_t n_cache_files = 0;
        for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            n_cache_files += CommonTestUtils::listFilesWithExt(cache_path, ext).size();
        }

        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(CommonTestUtils::directoryExists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Load 2nd CNNNetwork
        auto execNet2 = core->compile_model(function, targetDevice, configuration);
        execNet2 = {};
        size_t n_cache_files_compare = 0;
        // Check that two loaded networks with same function creates same caches
        for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            n_cache_files_compare += CommonTestUtils::listFilesWithExt(cache_path, ext).size();
            ASSERT_TRUE(CommonTestUtils::removeFilesWithExt(cache_path, ext));
        }
        ASSERT_EQ(n_cache_files_compare, n_cache_files);

        // Remove directory and check that it doesn't exist anymore
        ASSERT_EQ(CommonTestUtils::removeDir(cache_path), 0);
        ASSERT_FALSE(CommonTestUtils::directoryExists(cache_path));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (CommonTestUtils::directoryExists(cache_path)) {
            for (auto& ext : m_extList) {
                // Check that folder contains cache files and remove them
                ASSERT_GE(CommonTestUtils::removeFilesWithExt(cache_path, ext), 0);
            }
            ASSERT_EQ(CommonTestUtils::removeDir(cache_path), 0);
        }
        FAIL() << ex.what() << std::endl;
    }
}


#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

TEST_P(CompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinariesUnicodePath) {
    #if defined(_WIN32) || defined(_WIN64)
        GTEST_SKIP();
    #else
        for (std::size_t testIndex = 0; testIndex < CommonTestUtils::test_unicode_postfix_vector.size(); testIndex++) {
            std::wstring postfix  = L"_" + CommonTestUtils::test_unicode_postfix_vector[testIndex];
            std::wstring cache_path_w = CommonTestUtils::stringToWString(cache_path) + postfix;

            try {
                auto cache_path_mb = ov::util::wstring_to_string(cache_path_w);
                core->set_property(ov::cache_dir(cache_path_mb));
                // Load CNNNetwork to target plugins
                auto execNet = core->compile_model(function, targetDevice, configuration);
                execNet = {};
                // Check that directory with cached kernels exists after loading network
                ASSERT_TRUE(CommonTestUtils::directoryExists(cache_path_w)) << "Directory with cached kernels doesn't exist";
                // Check that folder contains cache files and remove them
                for (auto& ext : m_extList) {
                    // Check that folder contains cache files and remove them
                    ASSERT_GT(CommonTestUtils::removeFilesWithExt(cache_path_w, CommonTestUtils::stringToWString(ext)), 0);
                }
                // Remove directory and check that it doesn't exist anymore
                ASSERT_EQ(CommonTestUtils::removeDir(cache_path_w), 0);
                ASSERT_FALSE(CommonTestUtils::directoryExists(cache_path_w));
            } catch (std::exception& ex) {
                // Cleanup in case of any exception
                if (CommonTestUtils::directoryExists(cache_path_w)) {
                    for (auto& ext : m_extList) {
                        // Check that folder contains cache files and remove them
                        ASSERT_GT(CommonTestUtils::removeFilesWithExt(cache_path_w, CommonTestUtils::stringToWString(ext)), 0);
                    }
                    ASSERT_EQ(CommonTestUtils::removeDir(cache_path_w), 0);
                }
                FAIL() << ex.what() << std::endl;
            }
        }
    #endif
}
#endif
} // namespace behavior
} // namespace test
} // namespace ov
