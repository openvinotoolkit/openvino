// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using softMaxLayerTestParams = std::tuple<
        InferenceEngine::Precision,                          // netPrecision
        InferenceEngine::Precision,                          // Input precision
        InferenceEngine::Precision,                          // Output precision
        InferenceEngine::Layout,                             // Input layout
        InferenceEngine::Layout,                             // Output layout
        std::vector<std::vector<std::pair<size_t, size_t>>>, // Input shape
        std::vector<std::vector<std::vector<size_t>>>,       // Target shapes
        size_t,                                              // axis
        std::string,                                         // targetDevice
        std::map<std::string, std::string>                   // config
>;

class SoftMaxLayerTest : public testing::WithParamInterface<softMaxLayerTestParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<softMaxLayerTestParams>& obj);

protected:
    void SetUp() override;
    std::shared_ptr<ngraph::Function> makeSoftMax(const std::string& name = "");
    void setTargetStaticShape(std::vector<ngraph::Shape>& desiredTargetStaticShape) override;

private:
    InferenceEngine::Precision netPrecision;
    size_t axis;
};

}  // namespace LayerTestsDefinitions
