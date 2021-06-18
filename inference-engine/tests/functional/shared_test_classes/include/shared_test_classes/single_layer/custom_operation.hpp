// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Target device name
> CustomOpLayerParams;

class CustomOpLayerTest: public testing::WithParamInterface<CustomOpLayerParams>,
                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CustomOpLayerParams>& obj);

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

    CustomOpLayerTest();

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions

