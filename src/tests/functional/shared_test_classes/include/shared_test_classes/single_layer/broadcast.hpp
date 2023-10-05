// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using BroadcastParamsTuple = typename std::tuple<
        InferenceEngine::SizeVector,       // target shape
        ngraph::AxisSet,                   // axes mapping
        ngraph::op::BroadcastType,         // broadcast mode
        InferenceEngine::SizeVector,       // Input shape
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class BroadcastLayerTest : public testing::WithParamInterface<BroadcastParamsTuple>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
