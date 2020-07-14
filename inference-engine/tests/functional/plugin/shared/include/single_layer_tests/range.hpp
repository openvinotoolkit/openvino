// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        float,                          // start
        float,                          // stop
        float,                          // step
        InferenceEngine::Precision,     // Net precision
        std::string                     // Target device name
> RangeParams;

class RangeLayerTest : public testing::WithParamInterface<RangeParams>,
                       public LayerTestsUtils::LayerTestsCommon {
    float start, stop, step;
public:
    static std::string getTestCaseName(testing::TestParamInfo<RangeParams> obj);
    void Infer();

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions