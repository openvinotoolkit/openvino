// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <typeindex>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/node.hpp>
#include <ngraph/function.hpp>
#include <ie_plugin_config.hpp>
#include <ngraph/function.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include "gtest/gtest.h"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

namespace BehaviorTestsDefinitions {
    typedef std::tuple<
            InferenceEngine::Precision,             // Network precision
            std::string,                            // Target device name
            std::map<std::string, std::string>,     // Target config
            InferenceEngine::Layout,                // Layout
            std::vector<size_t>>                    // InputShapes
    LayoutParams;

class LayoutTest : public CommonTestUtils::TestsCommon,
        public ::testing::WithParamInterface<LayoutParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayoutParams> obj);

    void SetUp() override;

    void TearDown() override;

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    InferenceEngine::Layout layout;
    std::vector<size_t> inputShapes;
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
};
}  // namespace BehaviorTestsDefinitions