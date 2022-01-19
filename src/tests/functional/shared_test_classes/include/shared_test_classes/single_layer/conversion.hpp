// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <map>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_core.hpp"

namespace LayerTestsDefinitions {

static std::map<ngraph::helpers::ConversionTypes, std::string> conversionNames = {
    {ngraph::helpers::ConversionTypes::CONVERT, "Convert"},
    {ngraph::helpers::ConversionTypes::CONVERT_LIKE, "ConvertLike"}};

using ConversionParamsTuple = typename std::tuple<ngraph::helpers::ConversionTypes,  // Convertion op type
                                                  std::vector<std::vector<size_t>>,  // Input1 shapes
                                                  InferenceEngine::Precision,        // Input1 precision
                                                  InferenceEngine::Precision,        // Input2 precision
                                                  InferenceEngine::Layout,           // Input layout
                                                  InferenceEngine::Layout,           // Output layout
                                                  std::string>;                      // Device name

class ConversionLayerTest : public testing::WithParamInterface<ConversionParamsTuple>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConversionParamsTuple>& obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
