// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "constant.hpp"

namespace LayerTestsDefinitions {

struct RandomUniformTypeSpecificParams {
    InferenceEngine::Precision precision;  // Output data precision
    double min_value; // min value constant, will be cast to the needed precision
    double max_value; // max value constant, will be cast to the needed precision
};

using RandomUniformParamsTuple = typename std::tuple<
        ov::Shape, // output shape
        RandomUniformTypeSpecificParams, // parameters which depends on output type
        int64_t, // global seed
        int64_t, // operation seed
        std::string>; // Device name

class RandomUniformLayerTest : public testing::WithParamInterface<RandomUniformParamsTuple>,
                               virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUniformParamsTuple> &obj);

protected:
    void SetUp() override;

    void ConvertRefsParams() override;
};

}  // namespace LayerTestsDefinitions

