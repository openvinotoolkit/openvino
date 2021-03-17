// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <ie_common.h>
#include <thread>

#include "behavior/caching_tests.hpp"
#include "common_test_utils/file_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

using namespace InferenceEngine::details;
using namespace InferenceEngine;
using namespace ::testing;
using namespace std::placeholders;

#define GTEST_COUT std::cout << "[          ] [ INFO ] "

namespace LayerTestsDefinitions {

const std::vector<ngraph::element::Type> LoadNetworkCacheTestBase::precisions = {
    ngraph::element::f32,
    ngraph::element::f16,
    ngraph::element::bf16,
    ngraph::element::i32,
    ngraph::element::i16,
    ngraph::element::i8,
    ngraph::element::u32,
    ngraph::element::u16,
    ngraph::element::u8,
};

static std::shared_ptr<ngraph::Function> create_simple_function(ngraph::element::Type type) {
    // Create Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset6::Parameter>(type, ngraph::Shape{2, 2});
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

std::vector<nGraphFunctionWithName> LoadNetworkCacheTestBase::getStandardFunctions() {
    // Wrapper of most part of available builder functions
    using ngraphFunctionIS = std::function<std::shared_ptr<ngraph::Function>(std::vector<size_t> inputShape, ngraph::element::Type_t type)>;
    auto inputShapeWrapper = [](ngraphFunctionIS fun, std::vector<size_t> inputShape) {
        return [fun, inputShape](ngraph::element::Type type) {
            return fun(inputShape, type);
        };
    };

    std::vector<nGraphFunctionWithName> res;
    res.push_back({create_simple_function, "SimpleFunction"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeConvPoolRelu, {1, 1, 32, 32}),
                   "ConvPoolRelu"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcat, {1, 4, 20, 20}),
                   "SplitConvConcat"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeKSOFunction, {1, 4, 20, 20}),
                   "KSOFunction"});
    res.push_back({ngraph::builder::subgraph::makeTIwithLSTMcell,
                   "TIwithLSTMcell"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeSingleConv, {1, 3, 24, 24}),
                   "SingleConv"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::make2InputSubtract, {1, 3, 24, 24}),
                   "2InputSubtract"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeNestedSplitConvConcat, {1, 4, 20, 20}),
                   "NestedSplitConvConcat"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatInputInBranch, {1, 4, 20, 20}),
                   "SplitConvConcatInputInBranch"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatNestedInBranch, {1, 4, 20, 20}),
                   "SplitConvConcatNestedInBranch"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeSplitConvConcatNestedInBranchNestedOut, {1, 4, 20, 20}),
                   "SplitConvConcatNestedInBranchNestedOut"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeConvBias, {1, 3, 24, 24}),
                   "ConvBias"});
    res.push_back({inputShapeWrapper(ngraph::builder::subgraph::makeReadConcatSplitAssign, {1, 1, 2, 4}),
                   "ReadConcatSplitAssign"});

    return res;
}

bool LoadNetworkCacheTestBase::importExportSupported(InferenceEngine::Core& ie) const {
    std::vector<std::string> supportedMetricKeys = ie.GetMetric(targetDevice, METRIC_KEY(SUPPORTED_METRICS));
    auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(),
                        METRIC_KEY(IMPORT_EXPORT_SUPPORT));
    bool supported = (it != supportedMetricKeys.end()) &&
                     ie.GetMetric(targetDevice, METRIC_KEY(IMPORT_EXPORT_SUPPORT));
    return supported;
}

std::string LoadNetworkCacheTestBase::getTestCaseName(testing::TestParamInfo<loadNetworkCacheParams> obj) {
    auto param = obj.param;
    auto funcName = std::get<1>(std::get<0>(param));
    auto precision = std::get<1>(param);
    auto deviceName = std::get<2>(param);
    return funcName + "_" + ngraph::element::Type(precision).get_type_name() + "_" + deviceName;
}

void LoadNetworkCacheTestBase::SetUp() {
    nGraphFunctionWithName funcPair;
    std::tie(funcPair, m_precision, targetDevice) = GetParam();
    auto fGen = std::get<0>(funcPair);
    m_functionName = std::get<1>(funcPair);
    try {
        function = fGen(m_precision);
    } catch (...) {
        SKIP();
    }

    std::stringstream ss;
    auto hash = std::hash<std::string>()(GetTestName());
    ss << std::to_string(hash) << "_" << std::this_thread::get_id() << "_" << GetTimestamp();
    m_cacheFolderName = ss.str();
}

void LoadNetworkCacheTestBase::TearDown() {
    CommonTestUtils::removeFilesWithExt(m_cacheFolderName, "blob");
    std::remove(m_cacheFolderName.c_str());
}

void LoadNetworkCacheTestBase::Run() {
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
        GTEST_COUT << "Can't create function " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
        SKIP();
    }
    if (!importExportSupported(*core)) {
        GTEST_COUT << "Plugin doesn't support import and export - skipping test" << std::endl;
        SKIP();
    }
    cnnNetwork = CNNNetwork{function};
    ConfigureNetwork();
    try {
        core = std::make_shared<Core>();
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
        GenerateInputs();
        Infer();
    } catch (InferenceEngineException &ex) {
        GTEST_COUT << "Can't loadNetwork without cache for " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
        GTEST_COUT << "Exception [" << ex.what() << "]" << std::endl;
        SKIP();
    } catch (...) {
        GTEST_COUT << "Can't loadNetwork without cache for " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
        SKIP(); // skip caching test if such network is not supported by device at all
    }
    auto originalOutputs = GetOutputs();
    {
        core = std::make_shared<Core>();
        core->SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheFolderName}});
        ASSERT_NO_THROW(executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration));
        Infer();
    }
    // cache is created
    ASSERT_EQ(CommonTestUtils::listFilesWithExt(m_cacheFolderName, "blob").size(), 1);
    compareOutputs(originalOutputs, GetOutputs());
    {
        core = std::make_shared<Core>();
        core->SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheFolderName}});
        ASSERT_NO_THROW(executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration));
        Infer();
    }
    // no new cache is created
    ASSERT_EQ(CommonTestUtils::listFilesWithExt(m_cacheFolderName, "blob").size(), 1);
    compareOutputs(originalOutputs, GetOutputs());
}

TEST_P(LoadNetworkCacheTestBase, CompareWithRefImpl) {
    Run();
}

} // namespace LayerTestsDefinitions
