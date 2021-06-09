// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using ConfigMap = typename std::map<std::string, std::string>;

using ResultTestParamSet = typename std::tuple<
    InferenceEngine::SizeVector,          // Input shapes
    InferenceEngine::Precision,           // Input precision
    LayerTestsUtils::TargetDevice,        // Device name
    ConfigMap>;                           // Additional network configuration

class ResultLayerTest : public testing::WithParamInterface<ResultTestParamSet>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ResultTestParamSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
