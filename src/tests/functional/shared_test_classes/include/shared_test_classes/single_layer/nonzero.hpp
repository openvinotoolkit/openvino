// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

namespace LayerTestsDefinitions {

using ConfigMap = typename std::map<std::string, std::string>;

using NonZeroLayerTestParamsSet = typename std::tuple<
    InferenceEngine::SizeVector,          // Input shapes
    InferenceEngine::Precision,           // Input precision
    LayerTestsUtils::TargetDevice,        // Device name
    ConfigMap>;                           // Additional network configuration

class NonZeroLayerTest : public testing::WithParamInterface<NonZeroLayerTestParamsSet>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NonZeroLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
