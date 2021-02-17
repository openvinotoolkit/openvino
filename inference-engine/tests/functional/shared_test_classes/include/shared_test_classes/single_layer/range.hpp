// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        float,                          // start
        float,                          // stop
        float,                          // step
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        std::string                     // Target device name
> RangeParams;

class RangeLayerTest : public testing::WithParamInterface<RangeParams>,
                       virtual public LayerTestsUtils::LayerTestsCommon {
    float start, stop, step;
public:
    static std::string getTestCaseName(testing::TestParamInfo<RangeParams> obj);
    void Infer() override;

protected:
    void SetUp() override;
};

class RangeNumpyLayerTest : public testing::WithParamInterface<RangeParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RangeParams> obj);
    void Infer() override;
protected:
    void SetUp() override;
private:
    float start, stop, step;
};

}  // namespace LayerTestsDefinitions