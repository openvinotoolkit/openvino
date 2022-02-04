// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph/function.hpp"

#include <ie_core.hpp>
#include <ie_common.h>

using ngraphFunctionGenerator = std::function<std::shared_ptr<ngraph::Function>(ngraph::element::Type, std::size_t)>;
using nGraphFunctionWithName = std::tuple<ngraphFunctionGenerator, std::string>;

using loadNetworkCacheParams = std::tuple<
        nGraphFunctionWithName, // ngraph function with friendly name
        ngraph::element::Type,  // precision
        std::size_t,            // batch size
        std::string             // device name
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

} // namespace LayerTestsDefinitions
