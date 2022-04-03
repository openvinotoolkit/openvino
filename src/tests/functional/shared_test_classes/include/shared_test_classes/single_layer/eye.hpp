// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
using eyeParams = std::tuple<
        std::vector<size_t>,                // input shape
        std::vector<int>,                   // eye params (rows, cols, diag_shift)
        InferenceEngine::Precision,         // net precision
        LayerTestsUtils::TargetDevice>;     // device name

class EyeLayerTest : public testing::WithParamInterface<eyeParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
    void GenerateInputs() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<eyeParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
