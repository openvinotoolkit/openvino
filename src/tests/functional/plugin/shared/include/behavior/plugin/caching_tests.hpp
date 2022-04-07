// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph/function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/util/common_util.hpp"

#include <ie_core.hpp>
#include <ie_common.h>

using ngraphFunctionGenerator = std::function<std::shared_ptr<ngraph::Function>(ngraph::element::Type, std::size_t)>;
using nGraphFunctionWithName = std::tuple<ngraphFunctionGenerator, std::string>;

using loadNetworkCacheParams = std::tuple<
        nGraphFunctionWithName, // ngraph function with friendly name
        ngraph::element::Type,  // precision
        std::size_t,            // batch size
        std::string,            // device name
        std::map<std::string, std::string> //device configuration
        >;

namespace LayerTestsDefinitions {

class LoadNetworkCacheTestBase : public testing::WithParamInterface<loadNetworkCacheParams>,
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

    // Default functions and precisions that can be used as test parameters
    static std::vector<nGraphFunctionWithName> getStandardFunctions();
};

using compileKernelsCacheParams = std::tuple<
        std::string,            // device name
        std::map<std::string, std::string>    // device configuration
>;
class LoadNetworkCompiledKernelsCacheTest : virtual public LayerTestsUtils::LayerTestsCommon,
                                 public testing::WithParamInterface<compileKernelsCacheParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<compileKernelsCacheParams> obj);
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::shared_ptr<ngraph::Function> function;
    std::string cache_path;
    void SetUp() override {
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        std::tie(targetDevice, configuration) = GetParam();
        test_name.erase(remove(test_name.begin(), test_name.end(), '/'), test_name.end());
        cache_path = test_name + "_cache";
    }
};
using LoadNetworkCompileWithCacheNoThrowTest = LoadNetworkCompiledKernelsCacheTest;
} // namespace LayerTestsDefinitions
