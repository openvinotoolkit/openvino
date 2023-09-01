// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        double,                        // epsilon
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::Layout,       // Output layout
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Target device name
> BatchNormLayerTestParams;

class BatchNormLayerTest : public testing::WithParamInterface<BatchNormLayerTestParams>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchNormLayerTestParams>& obj);

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
