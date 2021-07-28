// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOTE: WILL BE REWORKED (31905)

#include <gtest/gtest.h>

#include <map>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_core.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    std::vector<std::vector<size_t>>,             // input shapes
    ngraph::helpers::EltwiseTypes,                // eltwise op type
    ngraph::helpers::InputLayerType,              // secondary input type
    CommonTestUtils::OpType,                      // op type
    InferenceEngine::Precision,                   // Net precision
    InferenceEngine::Precision,                   // Input precision
    InferenceEngine::Precision,                   // Output precision
    InferenceEngine::Layout,                      // Input layout
    std::string,                                  // Device name
    std::map<std::string, std::string>            // Additional network configuration
> EltwiseTestParams;

class EltwiseLayerTest : public testing::WithParamInterface<EltwiseTestParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    void SetUp() override;

public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseTestParams> obj);
};
} // namespace LayerTestsDefinitions
