// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <map>
#include <functional_test_utils/layer_test_utils.hpp>

#include "common_test_utils/common_layers_params.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_core.hpp"

namespace LayerTestsDefinitions {
namespace LogicalParams {
using InputShapesTuple = std::pair<std::vector<size_t>, std::vector<size_t>>;
} // LogicalParams

typedef std::tuple<
    LogicalParams::InputShapesTuple,    // Input shapes tuple
    InferenceEngine::Precision,         // Inputs precision
    ngraph::helpers::LogicalTypes,      // Logical op type
    ngraph::helpers::InputLayerType,    // Second input type
    InferenceEngine::Precision,         // Net precision
    std::string,                        // Device name
    std::map<std::string, std::string>  // Additional network configuration
> LogicalTestParams;

class LogicalLayerTest : public testing::WithParamInterface<LogicalTestParams>,
    public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(testing::TestParamInfo<LogicalTestParams> obj);
    static std::vector<LogicalParams::InputShapesTuple> combineShapes(const std::map<std::vector<size_t>, std::vector<std::vector<size_t >>>& inputShapes);
};
} // namespace LayerTestsDefinitions
