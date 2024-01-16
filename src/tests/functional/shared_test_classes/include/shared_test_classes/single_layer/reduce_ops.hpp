// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<int>,               // Axis to reduce order
        ov::test::utils::OpType,        // Scalar or vector type axis
        bool,                           // Keep dims
        ngraph::helpers::ReductionType, // Reduce operation type
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        std::vector<size_t>,            // Input shapes
        std::string                     // Target device name
> reduceMeanParams;

class ReduceOpsLayerTest : public testing::WithParamInterface<reduceMeanParams>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<reduceMeanParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

class ReduceOpsLayerWithSpecificInputTest : public ReduceOpsLayerTest {
protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
};

}  // namespace LayerTestsDefinitions
