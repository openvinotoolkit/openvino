// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <thread>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph/function.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/util/common_util.hpp"
#include "base/behavior_test_utils.hpp"

#include <ie_core.hpp>
#include <ie_common.h>

using ngraphFunctionGenerator = std::function<std::shared_ptr<ngraph::Function>(ngraph::element::Type, std::size_t)>;
using nGraphFunctionWithName = std::tuple<ngraphFunctionGenerator, std::string>;
using ngraphFunctionIS = std::function<std::shared_ptr<ngraph::Function>(std::vector<size_t> inputShape,
                                                                         ngraph::element::Type_t type)>;

using loadNetworkCacheParams = std::tuple<
        nGraphFunctionWithName, // ngraph function with friendly name
        ngraph::element::Type,  // precision
        std::size_t,            // batch size
        std::string            // device name
        >;

namespace LayerTestsDefinitions {

class LoadNetworkCacheTestBase : public testing::WithParamInterface<loadNetworkCacheParams>,
                                 virtual public BehaviorTestsUtils::IEPluginTestBase,
                                 virtual public LayerTestsUtils::LayerTestsCommon {
    std::string           m_cacheFolderName;
    std::string           m_functionName;
    ngraph::element::Type m_precision;
    size_t                m_batchSize;
public:
    static std::string getTestCaseName(testing::TestParamInfo<loadNetworkCacheParams> obj);
    void SetUp() override;
    void TearDown() override;
    void Run() override;

    bool importExportSupported(InferenceEngine::Core& ie) const;

    // Wrapper of most part of available builder functions
    static ngraphFunctionGenerator inputShapeWrapper(ngraphFunctionIS fun, std::vector<size_t> inputShape);
    // Default functions and precisions that can be used as test parameters
    static std::vector<nGraphFunctionWithName> getAnyTypeOnlyFunctions();
    static std::vector<nGraphFunctionWithName> getNumericTypeOnlyFunctions();
    static std::vector<nGraphFunctionWithName> getNumericAnyTypeFunctions();
    static std::vector<nGraphFunctionWithName> getFloatingPointOnlyFunctions();
    static std::vector<nGraphFunctionWithName> getStandardFunctions();
};

using compileKernelsCacheParams = std::tuple<
        std::string,            // device name
        std::pair<std::map<std::string, std::string>, std::string>   // device and cache configuration
>;

OPENVINO_DISABLE_WARNING_MSVC_BEGIN(4250)  // Visual Studio warns us about inheritance via dominance but it's done intentionally
                                           // so turn it off
class LoadNetworkCompiledKernelsCacheTest : virtual public LayerTestsUtils::LayerTestsCommon,
                                            virtual public BehaviorTestsUtils::IEPluginTestBase,
                                            public testing::WithParamInterface<compileKernelsCacheParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<compileKernelsCacheParams> obj);
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string cache_path;
    std::vector<std::string> m_extList;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::pair<std::map<std::string, std::string>, std::string> userConfig;
        std::tie(targetDevice, userConfig) = GetParam();
        target_device = targetDevice;
        APIBaseTest::SetUp();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
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
        cache_path = "LoadNetwork" + std::to_string(hash) + "_"
                + ss.str() + "_" + GetTimestamp() + "_cache";
    }
    void TearDown() override {
        APIBaseTest::TearDown();
    }
};

OPENVINO_DISABLE_WARNING_MSVC_END(4250)

} // namespace LayerTestsDefinitions
