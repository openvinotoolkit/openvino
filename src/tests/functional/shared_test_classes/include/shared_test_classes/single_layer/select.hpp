// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>,  // mask, then, else shapes
        InferenceEngine::Precision,        // then, else precision
        ngraph::op::AutoBroadcastSpec,     // broadcast
        std::string> selectTestParams;     // device name

class SelectLayerTest : public testing::WithParamInterface<selectTestParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo <selectTestParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
