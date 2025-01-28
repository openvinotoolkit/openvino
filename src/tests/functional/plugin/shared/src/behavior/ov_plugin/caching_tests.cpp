// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <gtest/gtest.h>
#include <thread>

#include "behavior/ov_plugin/caching_tests.hpp"

#include "openvino/pass/manager.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/summary/api_summary.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/kso_func.hpp"
#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"
#include "common_test_utils/subgraph_builders/single_conv.hpp"
#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"
#include "common_test_utils/subgraph_builders/nested_split_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/conv_bias.hpp"
#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"
#include "common_test_utils/subgraph_builders/matmul_bias.hpp"

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
    auto func = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data});
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
    return [fun, inputShape](ov::element::Type type, std::size_t batchSize) {
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
        inputShapeWrapper(ov::test::utils::make_conv_pool_relu, {1, 1, 32, 32}),
        "ConvPoolRelu"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_split_conv_concat, {1, 4, 20, 20}),
        "SplitConvConcat"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_kso_function, {1, 4, 20, 20}),
        "KSOFunction"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_single_conv, {1, 3, 24, 24}),
        "SingleConv"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_2_input_subtract, {1, 3, 24, 24}),
        "2InputSubtract"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_nested_split_conv_concat, {1, 4, 20, 20}),
        "NestedSplitConvConcat"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_cplit_conv_concat_input_in_branch, {1, 4, 20, 20}),
        "SplitConvConcatInputInBranch"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_cplit_conv_concat_nested_in_branch, {1, 4, 20, 20}),
        "SplitConvConcatNestedInBranch"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_cplit_conv_concat_nested_in_branch_nested_out, {1, 4, 20, 20}),
        "SplitConvConcatNestedInBranchNestedOut"});
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_conv_bias, {1, 3, 24, 24}),
        "ConvBias"});
    res.push_back(ovModelWithName{
        inputShapeWrapper(ov::test::utils::make_matmul_bias, {1, 3, 24, 24}),
        "MatMulBias" });
    return res;
}

std::vector<ovModelWithName> CompileModelCacheTestBase::getAnyTypeOnlyFunctions() {
    std::vector<ovModelWithName> res;
    res.push_back(ovModelWithName {
        inputShapeWrapper(ov::test::utils::make_read_concat_split_assign, {1, 1, 2, 4}),
        "ReadConcatSplitAssign"});
    return res;
}

std::vector<ovModelWithName> CompileModelCacheTestBase::getFloatingPointOnlyFunctions() {
    std::vector<ovModelWithName> res;
    res.push_back(ovModelWithName { [](ov::element::Type type, size_t batchSize) {
        return ov::test::utils::make_ti_with_lstm_cell(type, batchSize);
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
    return funcName + "_" + ov::element::Type(precision).get_type_name() + "_batch" + std::to_string(batchSize) + "_" + deviceName;
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
    ov::test::utils::PluginCache::get().reset();
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
        OV_ASSERT_NO_THROW(core->get_property(targetDevice, ov::internal::caching_properties));
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
    size_t blobCountInitial = -1;
    size_t blobCountAfterwards = -1;
    for (int i = 0; i < 2; i++) {
        // Step 2: Load with cache. Export or import shall not throw
        {
            core->set_property(ov::cache_dir(m_cacheFolderName));
            OV_ASSERT_NO_THROW(compiledModel = core->compile_model(function, targetDevice, configuration));
            if (targetDevice.find("AUTO") == std::string::npos) {
                // Apply check only for HW plugins
                ASSERT_EQ(i != 0, compiledModel.get_property(ov::loaded_from_cache));
            }
            generate_inputs(targetStaticShapes.front());
            OV_ASSERT_NO_THROW(infer());
        }
        compare(originalOutputs, get_plugin_outputs());
        // Destroy objects here
        // AUTO plugin will wait all HW plugins to finish compiling model
        // No impact for HW plugins
        compiledModel = {};
        inferRequest = {};
        if (i == 0) {
            // blob count should be greater than 0 initially
            blobCountInitial = ov::test::utils::listFilesWithExt(m_cacheFolderName, "blob").size();
            ASSERT_GT(blobCountInitial, 0);
        } else {
            // cache is created and reused. Blob count should be same as it was first time
            blobCountAfterwards = ov::test::utils::listFilesWithExt(m_cacheFolderName, "blob").size();
            ASSERT_EQ(blobCountInitial, blobCountAfterwards);
        }
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
    manager.run_passes(ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::f32));
}

void CompileModelLoadFromFileTestBase::TearDown() {
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "blob");
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "cl_cache");
    ov::test::utils::removeIRFiles(m_modelName, m_weightsName);
    std::remove(m_cacheFolderName.c_str());
    core->set_property(ov::cache_dir());
    ov::test::utils::PluginCache::get().reset();
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

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
TEST_P(CompileModelLoadFromFileTestBase, CanCreateCacheDirAndDumpBinariesUnicodePath) {
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    auto hash = std::hash<std::string>()(test_name);
    std::stringstream ss;
    ss << std::this_thread::get_id();
    std::string cache_path = ov::test::utils::getCurrentWorkingDir() + ov::util::FileTraits<char>::file_separator +
                             "compiledModel_" + std::to_string(hash) + "_" + ss.str() + "_" + GetTimestamp() + "_cache";
    std::wstring postfix = L"_" + ov::test::utils::test_unicode_postfix_vector[0];
    std::wstring cache_path_w = ov::util::string_to_wstring(cache_path) + postfix;
    auto cache_path_mb = ov::util::wstring_to_string(cache_path_w);
    std::wstring model_xml_path_w =
        ov::util::string_to_wstring(cache_path_mb + ov::util::FileTraits<char>::file_separator + m_modelName);
    std::wstring model_bin_path_w =
        ov::util::string_to_wstring(cache_path_mb + ov::util::FileTraits<char>::file_separator + m_weightsName);

    try {
        ov::test::utils::createDirectory(cache_path_w);

        // Copy IR files into unicode folder for read_model test
        ov::test::utils::copyFile(m_modelName, model_xml_path_w);
        ov::test::utils::copyFile(m_weightsName, model_bin_path_w);

        // Set unicode folder as cache_dir
        core->set_property(ov::cache_dir(cache_path_mb));
        // Read model from unicode folder
        auto model = core->read_model(ov::util::wstring_to_string(model_xml_path_w));

        // Load model to target plugins
        auto compiled_model = core->compile_model(model, targetDevice, configuration);
        compiled_model = {};
        model = {};
        // Check that directory with cached model exists after loading network
        ASSERT_TRUE(ov::util::directory_exists(cache_path_w)) << "Directory with cached kernels doesn't exist";
        // Check that folder contains cache files and remove them
        int removed_files_num = 0;
        removed_files_num += ov::test::utils::removeFilesWithExt(cache_path_w, ov::util::string_to_wstring("blob"));
        removed_files_num += ov::test::utils::removeFilesWithExt(cache_path_w, ov::util::string_to_wstring("cl_cache"));
        ASSERT_GT(removed_files_num, 0);
        ov::test::utils::removeFile(model_xml_path_w);
        ov::test::utils::removeFile(model_bin_path_w);
        // Remove directory and check that it doesn't exist anymore
        ov::test::utils::removeDir(cache_path_w);
        ASSERT_FALSE(ov::util::directory_exists(cache_path_w));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (ov::util::directory_exists(cache_path_w)) {
            // Check that folder contains cache files and remove them
            ASSERT_GT(ov::test::utils::removeFilesWithExt(cache_path_w, ov::util::string_to_wstring("blob")), 0);
            ov::test::utils::removeFile(model_xml_path_w);
            ov::test::utils::removeFile(model_bin_path_w);
            ov::test::utils::removeDir(cache_path_w);
        }
        FAIL() << ex.what() << std::endl;
    }
}
#endif

std::string CompileModelCacheRuntimePropertiesTestBase::getTestCaseName(
    testing::TestParamInfo<compileModelCacheRuntimePropertiesParams> obj) {
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

void CompileModelCacheRuntimePropertiesTestBase::SetUp() {
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
    manager.run_passes(ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::f32));
}

void CompileModelCacheRuntimePropertiesTestBase::TearDown() {
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "blob");
    ov::test::utils::removeIRFiles(m_modelName, m_weightsName);
    std::remove(m_cacheFolderName.c_str());
    core->set_property(ov::cache_dir());
    ov::test::utils::PluginCache::get().reset();
    APIBaseTest::TearDown();
}

void CompileModelCacheRuntimePropertiesTestBase::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    if (!ov::util::contains(core->get_property(target_device, ov::internal::supported_properties),
                            ov::internal::compiled_model_runtime_properties.name())) {
        return;
    }
    m_compiled_model_runtime_properties =
        core->get_property(target_device, ov::internal::compiled_model_runtime_properties);
    core->set_property(ov::cache_dir(m_cacheFolderName));

    // First compile model to generate model cache blob.
    // Second compile model will load from model cache.
    for (int i = 0; i < 2; i++) {
        {
            OV_ASSERT_NO_THROW(compiledModel = core->compile_model(m_modelName, targetDevice, configuration));
            ASSERT_EQ(i != 0, compiledModel.get_property(ov::loaded_from_cache));
            OV_ASSERT_NO_THROW(inferRequest = compiledModel.create_infer_request());
            OV_ASSERT_NO_THROW(inferRequest.infer());
        }
        // cache is created and reused
        ASSERT_EQ(ov::test::utils::listFilesWithExt(m_cacheFolderName, "blob").size(), 1);
        compiledModel = {};
        inferRequest = {};
    }

    // Modify cache blob file's header information to trigger removing old cache and to create new cache blob files.
    auto blobs = ov::test::utils::listFilesWithExt(m_cacheFolderName, "blob");
    for (const auto& fileName : blobs) {
        std::string content;
        {
            std::ifstream inp(fileName, std::ios_base::binary);
            std::ostringstream ostr;
            ostr << inp.rdbuf();
            content = ostr.str();
        }
        auto index = content.find(m_compiled_model_runtime_properties.c_str());
        ASSERT_EQ(index != std::string::npos, true);
        auto pos = m_compiled_model_runtime_properties.find(":");
        if (index != std::string::npos) {
            m_compiled_model_runtime_properties.replace(pos + 1, 1, "x");
        } else {
            m_compiled_model_runtime_properties.replace(1, 1, "x");
        }
        content.replace(index, m_compiled_model_runtime_properties.size(), m_compiled_model_runtime_properties);
        std::ofstream out(fileName, std::ios_base::binary);
        out.write(content.c_str(), static_cast<std::streamsize>(content.size()));
    }

    // Third compile model to remove old cache blob and create new model cache blob file
    // Fourth compile model will load from model cache.
    for (int i = 0; i < 2; i++) {
        {
            OV_ASSERT_NO_THROW(compiledModel = core->compile_model(m_modelName, targetDevice, configuration));
            ASSERT_EQ(i != 0, compiledModel.get_property(ov::loaded_from_cache));
            OV_ASSERT_NO_THROW(inferRequest = compiledModel.create_infer_request());
            OV_ASSERT_NO_THROW(inferRequest.infer());
        }
        // old cache has been removed and new cache is created and reused
        ASSERT_EQ(ov::test::utils::listFilesWithExt(m_cacheFolderName, "blob").size(), 1);
        compiledModel = {};
        inferRequest = {};
    }
}

TEST_P(CompileModelCacheRuntimePropertiesTestBase, CanLoadFromFileWithoutException) {
    run();
}

std::string CompileModelLoadFromCacheTest::getTestCaseName(
    testing::TestParamInfo<CompileModelLoadFromCacheParams> obj) {
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

void CompileModelLoadFromCacheTest::SetUp() {
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
    manager.run_passes(ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::f32));
}

void CompileModelLoadFromCacheTest::TearDown() {
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "blob");
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "cl_cache");
    ov::test::utils::removeIRFiles(m_modelName, m_weightsName);
    std::remove(m_cacheFolderName.c_str());
    core->set_property(ov::cache_dir());
    ov::test::utils::PluginCache::get().reset();
    APIBaseTest::TearDown();
}

void CompileModelLoadFromCacheTest::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    core->set_property(ov::cache_dir(m_cacheFolderName));
    compiledModel = core->compile_model(m_modelName, targetDevice, configuration);
    EXPECT_EQ(false, compiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());

    std::stringstream strm;
    compiledModel.export_model(strm);
    ov::CompiledModel importedCompiledModel = core->import_model(strm, target_device, configuration);
    EXPECT_EQ(false, importedCompiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());

    compiledModel = core->compile_model(m_modelName, targetDevice, configuration);
    EXPECT_EQ(true, compiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());
}

TEST_P(CompileModelLoadFromCacheTest, CanGetCorrectLoadedFromCacheProperty) {
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
    manager.run_passes(ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::f32));

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
    ov::test::utils::PluginCache::get().reset();
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
    function = ov::test::utils::make_conv_pool_relu();
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
    ov::test::utils::PluginCache::get().reset();
    APIBaseTest::TearDown();
}

TEST_P(CompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinaries) {
    core->set_property(ov::cache_dir(cache_path));
    try {
        // Load CNNNetwork to target plugins
        auto execNet = core->compile_model(function, targetDevice, configuration);
        execNet = {};
        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(ov::util::directory_exists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Check that folder contains cache files and remove them
        int number_of_deleted_files = 0;
        for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            number_of_deleted_files += ov::test::utils::removeFilesWithExt(cache_path, ext);
        }
        ASSERT_GT(number_of_deleted_files, 0);
        // Remove directory and check that it doesn't exist anymore
        ASSERT_EQ(ov::test::utils::removeDir(cache_path), 0);
        ASSERT_FALSE(ov::util::directory_exists(cache_path));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (ov::util::directory_exists(cache_path)) {
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
        ASSERT_TRUE(ov::util::directory_exists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Load 2nd CNNNetwork
        auto execNet2 = core->compile_model(function, targetDevice, configuration);
        execNet2 = {};
        size_t n_cache_files_compare = 0;
        int number_of_deleted_files = 0;
        // Check that two loaded networks with same function creates same caches
        for (auto& ext : m_extList) {
            // Check that folder contains cache files and remove them
            n_cache_files_compare += ov::test::utils::listFilesWithExt(cache_path, ext).size();
            number_of_deleted_files += ov::test::utils::removeFilesWithExt(cache_path, ext);
        }
        ASSERT_GT(number_of_deleted_files, 0);
        ASSERT_EQ(n_cache_files_compare, n_cache_files);

        // Remove directory and check that it doesn't exist anymore
        ASSERT_EQ(ov::test::utils::removeDir(cache_path), 0);
        ASSERT_FALSE(ov::util::directory_exists(cache_path));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (ov::util::directory_exists(cache_path)) {
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
            ASSERT_TRUE(ov::util::directory_exists(cache_path_w)) << "Directory with cached kernels doesn't exist";
            // Check that folder contains cache files and remove them
            int count_of_removed_files = 0;
            for (auto& ext : m_extList) {
                // Check that folder contains cache files and remove them
                count_of_removed_files += ov::test::utils::removeFilesWithExt(cache_path_w, ov::test::utils::stringToWString(ext));
            }
            ASSERT_GT(count_of_removed_files, 0);
            // Remove directory and check that it doesn't exist anymore
            ASSERT_EQ(ov::test::utils::removeDir(cache_path_w), 0);
            ASSERT_FALSE(ov::util::directory_exists(cache_path_w));
        } catch (std::exception& ex) {
            // Cleanup in case of any exception
            if (ov::util::directory_exists(cache_path_w)) {
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

std::string CompileModelWithCacheEncryptionTest::getTestCaseName(
    testing::TestParamInfo<std::string> obj) {
    auto deviceName = obj.param;
    std::ostringstream result;
    std::replace(deviceName.begin(), deviceName.end(), ':', '.');
    result << "device_name=" << deviceName << "_";
    return result.str();
}

void CompileModelWithCacheEncryptionTest::SetUp() {
    ovModelWithName funcPair;
    targetDevice = GetParam();
    target_device = targetDevice;
    EncryptionCallbacks encryption_callbacks;
    encryption_callbacks.encrypt = ov::util::codec_xor;
    encryption_callbacks.decrypt = ov::util::codec_xor;
    configuration.insert(ov::cache_encryption_callbacks(encryption_callbacks));
    APIBaseTest::SetUp();
    std::stringstream ss;
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    ss << "testCache_" << filePrefix;
    m_modelName = ss.str() + ".xml";
    m_weightsName = ss.str() + ".bin";
    m_cacheFolderName = ss.str();
    core->set_property(ov::cache_dir());
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(m_modelName, m_weightsName);
    manager.run_passes(ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::f32));
}

void CompileModelWithCacheEncryptionTest::TearDown() {
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "blob");
    ov::test::utils::removeFilesWithExt(m_cacheFolderName, "cl_cache");
    ov::test::utils::removeIRFiles(m_modelName, m_weightsName);
    std::remove(m_cacheFolderName.c_str());
    core->set_property(ov::cache_dir());
    ov::test::utils::PluginCache::get().reset();
    APIBaseTest::TearDown();
}

void CompileModelWithCacheEncryptionTest::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    core->set_property(ov::cache_dir(m_cacheFolderName));
    try {
        compiledModel = core->compile_model(m_modelName, targetDevice, configuration);
        EXPECT_EQ(false, compiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());

        std::stringstream strm;
        compiledModel.export_model(strm);
        ov::CompiledModel importedCompiledModel = core->import_model(strm, target_device, configuration);
        EXPECT_EQ(false, importedCompiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());

        compiledModel = core->compile_model(m_modelName, targetDevice, configuration);
        EXPECT_EQ(true, compiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());
    } catch (const Exception& ex) {
        GTEST_FAIL() << "Can't compile network from cache dir " << m_cacheFolderName <<
        "\nException [" << ex.what() << "]" << std::endl;
    } catch (...) {
        GTEST_FAIL() << "Can't compile network with model path " << m_modelName << std::endl;
    }
}

TEST_P(CompileModelWithCacheEncryptionTest, CanImportModelWithoutException) {
    run();
}
} // namespace behavior
} // namespace test
} // namespace ov
