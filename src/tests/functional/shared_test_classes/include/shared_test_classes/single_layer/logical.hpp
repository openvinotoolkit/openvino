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
namespace LogicalParams {
using InputShapesTuple = std::pair<std::vector<size_t>, std::vector<size_t>>;
} // LogicalParams

typedef std::tuple<
    LogicalParams::InputShapesTuple,    // Input shapes tuple
    ngraph::helpers::LogicalTypes,      // Logical op type
    ngraph::helpers::InputLayerType,    // Second input type
    InferenceEngine::Precision,         // Net precision
    InferenceEngine::Precision,         // Input precision
    InferenceEngine::Precision,         // Output precision
    InferenceEngine::Layout,            // Input layout
    InferenceEngine::Layout,            // Output layout
    std::string,                        // Device name
    std::map<std::string, std::string>  // Additional network configuration
> LogicalTestParams;

class LogicalLayerTest : public testing::WithParamInterface<LogicalTestParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    void SetupParams();
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<LogicalTestParams>& obj);
    static std::vector<LogicalParams::InputShapesTuple> combineShapes(const std::map<std::vector<size_t>, std::vector<std::vector<size_t >>>& inputShapes);

protected:
    LogicalParams::InputShapesTuple inputShapes;
    ngraph::helpers::LogicalTypes logicalOpType;
    ngraph::helpers::InputLayerType secondInputType;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
};
} // namespace LayerTestsDefinitions
