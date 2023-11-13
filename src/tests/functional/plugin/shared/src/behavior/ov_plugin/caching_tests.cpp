// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <gtest/gtest.h>
#include <thread>

#include "behavior/ov_plugin/caching_tests.hpp"

#include "openvino/pass/manager.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/summary/api_summary.hpp"

#include "ov_models/builders.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"

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

ovModelGenerator CompileModelCacheTestBase::inputShapeWrapper(ovModelIS fun, std::vector<size_t> inputShape) {
    return [fun, inputShape](ngraph::element::Type type, std::size_t batchSize) {
        auto shape = inputShape;
        shape[0] = batchSize;
        return fun(shape, type);
    };
}

std::vector<ovModelWithName> CompileModelCacheTestBase::getNumericTypeOnlyFunctions() {
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
    res.push_back(ovModelWithName{
        inputShapeWrapper(ngraph::builder::subgraph::makeMatMulBias, {1, 3, 24, 24}),
        "MatMulBias" });
    return res;
}

std::vector<ovModelWithName> CompileModelCacheTestBase::getAnyTypeOnlyFunctions() {
    std::vector<ovModelWithName> res;
    res.push_back(ovModelWithName {
        inputShapeWrapper(ngraph::builder::subgraph::makeReadConcatSplitAssign, {1, 1, 2, 4}),
        "ReadConcatSplitAssign"});
    return res;
}

std::vector<ovModelWithName> CompileModelCacheTestBase::getFloatingPointOnlyFunctions() {
    std::vector<ovModelWithName> res;
    res.push_back(ovModelWithName { [](ngraph::element::Type type, size_t batchSize) {
        return ngraph::builder::subgraph::makeTIwithLSTMcell(type, batchSize);
    }, "TIwithLSTMcell1"});
    return res;
}

std::vector<ovModelWithName> CompileModelCacheTestBase::getNumericAnyTypeFunctions() {
    std::vector<ovModelWithName> funcs = CompileModelCacheTestBase::getAnyTypeOnlyFunctions();
    std::vector<ovModelWithName> numericType = CompileModelCacheTestBase::getNumericTypeOnlyFunctions();
    funcs.insert(funcs.end(), numericType.begin(), numericType.end());

    return funcs;
}

std::vector<ovModelWithName> CompileModelCacheTestBase::getStandardFunctions() {
    std::vector<ovModelWithName> funcs = CompileModelCacheTestBase::getAnyTypeOnlyFunctions();
    std::vector<ovModelWithName> numericType = CompileModelCacheTestBase::getNumericTypeOnlyFunctions();
    funcs.insert(funcs.end(), numericType.begin(), numericType.end());
    std::vector<ovModelWithName> floatType = CompileModelCacheTestBase::getFloatingPointOnlyFunctions();
    funcs.insert(funcs.end(), floatType.begin(), floatType.end());

    return funcs;
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
    std::replace(deviceName.begin(), deviceName.end(), ':', '.');
    return funcName + "_" + ngraph::element::Type(precision).get_type_name() + "_batch" + std::to_string(batchSize) + "_" + deviceName;
}

void CompileModelCacheTestBase::SetUp() {
    ovModelWithName funcPair;
    std::tie(funcPair, m_precision, m_batchSize, targetDevice, configuration) = GetParam();
    target_device = targetDevice;
    APIBaseTest::SetUp();
    auto fGen = std::get<0>(funcPair);
    m_functionName = std::get<1>(funcPair);
    function = fGen(m_precision, m_batchSize);

    std::stringstream ss;
    auto hash = std::hash<std::string>()(SubgraphBaseTest::GetTestName());
    ss << "testCache_" << std::to_string(hash) << "_" << std::this_thread::get_id() << "_" << GetTimestamp();
    for (auto& iter : configuration) {
        ss << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
    }
    m_cacheFolderName = ss.str();
    core->set_property(ov::cache_dir());
}

void CompileModelCacheTestBase::TearDown() {
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "blob");
    std::remove(m_cacheFolderName.c_str());
    core->set_property(ov::cache_dir());
    try {
        core->set_property(targetDevice, ov::cache_dir());
    } catch (...) {
       // do nothing
    }
    APIBaseTest::TearDown();
}

void CompileModelCacheTestBase::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    if (!function) {
        GTEST_FAIL() << "Can't create function " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
    } else {
        std::vector<ov::Shape> inShapes;
        for (const auto& param : function->get_parameters()) {
            inShapes.push_back(param->get_shape());
        }
        init_input_shapes(static_shapes_to_test_representation(inShapes));
    }
    if ((targetDevice.find("GPU") != std::string::npos)) {
#if !defined(_WIN32) && !defined(_WIN64)
        setenv("OV_GPU_CACHE_MODEL", "1", 1);
#endif
    }
    if ((targetDevice.find("AUTO") == std::string::npos) && !importExportSupported(*core)) {
        GTEST_FAIL() << "Plugin doesn't support import and export - skipping test" << std::endl;
    }
    if (importExportSupported(*core)) {
        ASSERT_NO_THROW(core->get_property(targetDevice, ov::internal::caching_properties));
    }
    configure_model();
    try {
        compiledModel = core->compile_model(function, targetDevice, configuration);
        ASSERT_FALSE(compiledModel.get_property(ov::loaded_from_cache));
        generate_inputs(targetStaticShapes.front());
        infer();
    } catch (const Exception &ex) {
        GTEST_FAIL() << "Can't loadNetwork without cache for " << m_functionName << " with precision " << m_precision.get_type_name() <<
        "\nException [" << ex.what() << "]" << std::endl;
    } catch (...) {
        GTEST_FAIL() << "Can't compile network without cache for " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
    }
    auto originalOutputs = get_plugin_outputs();

    for (int i = 0; i < 2; i++) {
        // Step 2: Load with cache. Export or import shall not throw
        compiledModel = {}; // Destroy network object
        inferRequest = {};
        {
            core->set_property(ov::cache_dir(m_cacheFolderName));
            ASSERT_NO_THROW(compiledModel = core->compile_model(function, targetDevice, configuration));
            if (targetDevice.find("AUTO") == std::string::npos)
                // Apply check only for HW plugins
                ASSERT_EQ(i != 0, compiledModel.get_property(ov::loaded_from_cache));
            generate_inputs(targetStaticShapes.front());
            ASSERT_NO_THROW(infer());
        }
        // cache is created and reused
        ASSERT_EQ(ov::test::utils::listFilesWithExt(m_cacheFolderName, "blob").size(), 1);
        compare(originalOutputs, get_plugin_outputs());
    }
    if ((targetDevice.find("GPU") != std::string::npos)) {
#if !defined(_WIN32) && !defined(_WIN64)
        setenv("OV_GPU_CACHE_MODEL", "", 1);
#endif
    }
}

TEST_P(CompileModelCacheTestBase, CompareWithRefImpl) {
    run();
}

std::string CompileModelLoadFromFileTestBase::getTestCaseName(testing::TestParamInfo<compileModelLoadFromFileParams> obj) {
    auto param = obj.param;
    auto deviceName = std::get<0>(param);
    auto configuration = std::get<1>(param);
    std::ostringstream result;
    std::replace(deviceName.begin(), deviceName.end(), ':', '.');
    result << "device_name=" << deviceName << "_";
    for (auto& iter : configuration) {
        result << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
    }
    return result.str();
}

void CompileModelLoadFromFileTestBase::SetUp() {
    ovModelWithName funcPair;
    std::tie(targetDevice, configuration) = GetParam();
    target_device = targetDevice;
    APIBaseTest::SetUp();
    std::stringstream ss;
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    ss << "testCache_" << filePrefix;
    m_modelName = ss.str() + ".xml";
    m_weightsName = ss.str() + ".bin";
    for (auto& iter : configuration) {
        ss << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
    }
    m_cacheFolderName = ss.str();
    core->set_property(ov::cache_dir());
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(m_modelName, m_weightsName);
    manager.run_passes(ngraph::builder::subgraph::makeConvPoolRelu(
            {1, 3, 227, 227}, InferenceEngine::details::convertPrecision(InferenceEngine::Precision::FP32)));
}

void CompileModelLoadFromFileTestBase::TearDown() {
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "blob");
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "cl_cache");
    ov::test::utils::removeIRFiles(m_modelName, m_weightsName);
    std::remove(m_cacheFolderName.c_str());
    core->set_property(ov::cache_dir());
    APIBaseTest::TearDown();
}

void CompileModelLoadFromFileTestBase::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    core->set_property(ov::cache_dir(m_cacheFolderName));
    try {
        compiledModel = core->compile_model(m_modelName, targetDevice, configuration);
        inferRequest = compiledModel.create_infer_request();
        inferRequest.infer();
    } catch (const Exception &ex) {
        GTEST_FAIL() << "Can't loadNetwork with model path " << m_modelName <<
        "\nException [" << ex.what() << "]" << std::endl;
    } catch (...) {
        GTEST_FAIL() << "Can't compile network with model path " << m_modelName << std::endl;
    }
}

TEST_P(CompileModelLoadFromFileTestBase, CanLoadFromFileWithoutException) {
    run();
}

std::string CompileModelLoadFromMemoryTestBase::getTestCaseName(
    testing::TestParamInfo<compileModelLoadFromMemoryParams> obj) {
    auto param = obj.param;
    auto deviceName = std::get<0>(param);
    auto configuration = std::get<1>(param);
    std::ostringstream result;
    std::replace(deviceName.begin(), deviceName.end(), ':', '.');
    result << "device_name=" << deviceName << "_";
    for (auto& iter : configuration) {
        result << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
    }
    return result.str();
}

bool CompileModelLoadFromMemoryTestBase::importExportSupported(ov::Core& core) const {
    auto supportedProperties = core.get_property(targetDevice, ov::supported_properties);
    if (std::find(supportedProperties.begin(), supportedProperties.end(), ov::device::capabilities) ==
        supportedProperties.end()) {
        return false;
    }
    auto device_capabilities = core.get_property(targetDevice, ov::device::capabilities);
    if (std::find(device_capabilities.begin(),
                  device_capabilities.end(),
                  std::string(ov::device::capability::EXPORT_IMPORT)) == device_capabilities.end()) {
        return false;
    }
    return true;
}

void CompileModelLoadFromMemoryTestBase::SetUp() {
    ovModelWithName funcPair;
    std::tie(targetDevice, configuration) = GetParam();
    target_device = targetDevice;
    if ((targetDevice.find("GPU") != std::string::npos)) {
#if !defined(_WIN32) && !defined(_WIN64)
        setenv("OV_GPU_CACHE_MODEL", "1", 1);
#endif
    }
    APIBaseTest::SetUp();
    std::stringstream ss;
    auto hash = std::hash<std::string>()(SubgraphBaseTest::GetTestName());
    ss << "testCache_" << std::to_string(hash) << "_" << std::this_thread::get_id() << "_" << GetTimestamp();
    m_modelName = ss.str() + ".xml";
    m_weightsName = ss.str() + ".bin";
    for (auto& iter : configuration) {
        ss << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
    }
    m_cacheFolderName = ss.str();
    core->set_property(ov::cache_dir());
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(m_modelName, m_weightsName);
    manager.run_passes(ngraph::builder::subgraph::makeConvPoolRelu(
        {1, 3, 227, 227},
        InferenceEngine::details::convertPrecision(InferenceEngine::Precision::FP32)));

    try {
        std::ifstream model_file(m_modelName, std::ios::binary);
        std::stringstream ss;
        ss << model_file.rdbuf();
        m_model = ss.str();
    } catch (const Exception& ex) {
        GTEST_FAIL() << "Can't read xml file from: " << m_modelName << "\nException [" << ex.what() << "]" << std::endl;
    }

    try {
        std::ifstream weights_file(m_weightsName, std::ios::binary);
        weights_file.unsetf(std::ios::skipws);

        weights_file.seekg(0, std::ios::end);
        const auto weights_size = static_cast<std::size_t>(weights_file.tellg());
        weights_file.seekg(0, std::ios::beg);

        weights_vector.reserve(weights_size);
        weights_vector.insert(weights_vector.begin(),
                              std::istream_iterator<std::uint8_t>(weights_file),
                              std::istream_iterator<std::uint8_t>());
        m_weights = ov::Tensor(ov::element::u8, {1, 1, 1, weights_size}, weights_vector.data());
    } catch (const Exception& ex) {
        GTEST_FAIL() << "Can't read weights file from: " << m_weightsName << "\nException [" << ex.what() << "]"
                     << std::endl;
    }
}

void CompileModelLoadFromMemoryTestBase::TearDown() {
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "blob");
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "cl_cache");
    ov::test::utils::removeIRFiles(m_modelName, m_weightsName);
    std::remove(m_cacheFolderName.c_str());
    core->set_property(ov::cache_dir());
    APIBaseTest::TearDown();
    weights_vector.clear();
    if ((targetDevice.find("GPU") != std::string::npos)) {
#if !defined(_WIN32) && !defined(_WIN64)
        setenv("OV_GPU_CACHE_MODEL", "", 1);
#endif
    }
}

void CompileModelLoadFromMemoryTestBase::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    core->set_property(ov::cache_dir(m_cacheFolderName));
    for (int i = 0; i < 2; i++) {
        try {
            compiledModel = core->compile_model(m_model, m_weights, targetDevice, configuration);
            if (importExportSupported(*core)) {
                ASSERT_EQ(i != 0, compiledModel.get_property(ov::loaded_from_cache));
            }
            inferRequest = compiledModel.create_infer_request();
            inferRequest.infer();
        } catch (const Exception& ex) {
            GTEST_FAIL() << "Can't loadNetwork with model path " << m_modelName << "\nException [" << ex.what() << "]"
                         << std::endl;
        } catch (...) {
            GTEST_FAIL() << "Can't compile network with model path " << m_modelName << std::endl;
        }

        // For GPU plugin, KEY_GPU_THROUGHPUT_STREAMS will lead to config.throughput_streams==2, and Export stops.
        if (targetDevice.find("GPU") != std::string::npos) {
            auto item = configuration.find(ov::hint::performance_mode.name());
            if (item != configuration.end() &&
                item->second.as<ov::hint::PerformanceMode>() == ov::hint::PerformanceMode::THROUGHPUT) {
                break;
            }
        }
    }
}

TEST_P(CompileModelLoadFromMemoryTestBase, CanLoadFromMemoryWithoutExecption) {
    run();
}

TEST_P(CompileModelLoadFromMemoryTestBase, CanLoadFromMemoryWithoutWeightsANdExecption) {
    ov::pass::Manager manager;
    std::shared_ptr<ov::Model> model;
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});

        auto mul = std::make_shared<ov::op::v1::Multiply>(data, data);

        auto res = std::make_shared<ov::op::v0::Result>(mul);
        model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data});
    }

    manager.register_pass<ov::pass::Serialize>(m_modelName, m_weightsName);
    manager.run_passes(model);

    try {
        std::ifstream model_file(m_modelName, std::ios::binary);
        std::stringstream ss;
        ss << model_file.rdbuf();
        m_model = ss.str();
    } catch (const Exception& ex) {
        GTEST_FAIL() << "Can't read xml file from: " << m_modelName << "\nException [" << ex.what() << "]" << std::endl;
    }
    m_weights = ov::Tensor();
    run();
}

std::string CompiledKernelsCacheTest::getTestCaseName(testing::TestParamInfo<compileKernelsCacheParams> obj) {
    auto param = obj.param;
    std::string deviceName;
    std::pair<ov::AnyMap, std::string> userConfig;
    std::tie(deviceName, userConfig) = obj.param;
    std::replace(deviceName.begin(), deviceName.end(), ':', '.');
    auto properties = userConfig.first;
    std::ostringstream result;
    result << "device_name=" << deviceName << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    result << userConfig.second;
    return result.str();
}

void CompiledKernelsCacheTest::SetUp() {
    function = ngraph::builder::subgraph::makeConvPoolRelu();
    std::pair<ov::AnyMap, std::string> userConfig;
    std::tie(targetDevice, userConfig) = GetParam();
    target_device = targetDevice;
    APIBaseTest::SetUp();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    configuration = userConfig.first;
    std::string ext = userConfig.second;
    std::string::size_type pos = 0;
    if ((pos = ext.find(",", pos)) != std::string::npos) {
        m_extList.push_back(ext.substr(0, pos));
        m_extList.push_back(ext.substr(pos + 1));
    } else {
        m_extList.push_back(ext);
    }
    auto hash = std::hash<std::string>()(test_name);
    std::stringstream ss;
    ss << std::this_thread::get_id();
    cache_path = "compiledModel" + std::to_string(hash) + "_"
                + ss.str() + "_" + GetTimestamp() + "_cache";
}

void CompiledKernelsCacheTest::TearDown() {
    std::remove(cache_path.c_str());
    core->set_property(ov::cache_dir());
    APIBaseTest::TearDown();
}

TEST_P(CompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinaries) {
    core->set_property(ov::cache_dir(cache_path));
    try {
        // Load CNNNetwork to target plugins
        auto execNet = core->compile_model(function, targetDevice, configuration);
        execNet = {};
        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(ov::test::utils::directoryExists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Check that folder contains cache files and remove them
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
                ASSERT_GT(ov::test::utils::removeFilesWithExt(cache_path, ext), 0);
        }
            ASSERT_EQ(ov::test::utils::removeDir(cache_path), 0);
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
            n_cache_files += ov::test::utils::listFilesWithExt(cache_path, ext).size();
        }

        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(ov::test::utils::directoryExists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Load 2nd CNNNetwork
        auto execNet2 = core->compile_model(function, targetDevice, configuration);
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

TEST_P(CompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinariesUnicodePath) {
    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix  = L"_" + ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring cache_path_w = ov::test::utils::stringToWString(cache_path) + postfix;

        try {
            auto cache_path_mb = ov::util::wstring_to_string(cache_path_w);
            core->set_property(ov::cache_dir(cache_path_mb));
            // Load CNNNetwork to target plugins
            auto execNet = core->compile_model(function, targetDevice, configuration);
            execNet = {};
            // Check that directory with cached kernels exists after loading network
            ASSERT_TRUE(ov::test::utils::directoryExists(cache_path_w)) << "Directory with cached kernels doesn't exist";
            // Check that folder contains cache files and remove them
            for (auto& ext : m_extList) {
                // Check that folder contains cache files and remove them
                ASSERT_GT(ov::test::utils::removeFilesWithExt(cache_path_w, ov::test::utils::stringToWString(ext)), 0);
            }
            // Remove directory and check that it doesn't exist anymore
            ASSERT_EQ(ov::test::utils::removeDir(cache_path_w), 0);
            ASSERT_FALSE(ov::test::utils::directoryExists(cache_path_w));
        } catch (std::exception& ex) {
            // Cleanup in case of any exception
            if (ov::test::utils::directoryExists(cache_path_w)) {
                for (auto& ext : m_extList) {
                    // Check that folder contains cache files and remove them
                    ASSERT_GT(ov::test::utils::removeFilesWithExt(cache_path_w, ov::test::utils::stringToWString(ext)), 0);
                }
                ASSERT_EQ(ov::test::utils::removeDir(cache_path_w), 0);
            }
            FAIL() << ex.what() << std::endl;
        }
    }
}
#endif
} // namespace behavior
} // namespace test
} // namespace ov
