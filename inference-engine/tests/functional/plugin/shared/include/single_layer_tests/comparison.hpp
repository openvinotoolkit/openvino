// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <map>
#include <functional_test_utils/layer_test_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_core.hpp"

namespace LayerTestsDefinitions {
namespace ComparisonParams {
using InputShapesTuple = std::pair<std::vector<size_t>, std::vector<size_t>>;
} // ComparisonParams

typedef std::tuple<
    ComparisonParams::InputShapesTuple, // Input shapes tuple
    InferenceEngine::Precision,         // Inputs precision
    ngraph::helpers::ComparisonTypes,   // Comparison op type
    ngraph::helpers::InputLayerType,    // Second input type
    InferenceEngine::Precision,         // Net precision
    std::string,                        // Device name
    std::map<std::string, std::string>  // Additional network configuration
> ComparisonTestParams;

class ComparisonLayerTest : public testing::WithParamInterface<ComparisonTestParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ComparisonTestParams> obj);
};
} // namespace LayerTestsDefinitions
