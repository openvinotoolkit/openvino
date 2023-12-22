// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
using reverseParams = std::tuple<std::vector<size_t>,             // input shape
                                 std::vector<int>,                // axes
                                 std::string,                     // mode
                                 InferenceEngine::Precision,      // net precision
                                 LayerTestsUtils::TargetDevice>;  // device name

class ReverseLayerTest : public testing::WithParamInterface<reverseParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<reverseParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
