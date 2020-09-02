// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class PermuteTransformationTestValues {
public:
    class Actual {
    public:
        std::vector<float> fqInputLowIntervals;
        std::vector<float> fqInputHighIntervals;
        std::vector<float> fqOutputLowIntervals;
        std::vector<float> fqOutputHighIntervals;
    };

    class Expected {
    public:
        InferenceEngine::Precision permutePrecision;
        bool scales;
        bool shifts;
    };

    InferenceEngine::details::LayerTransformation::Params params;
    InferenceEngine::SizeVector inputShape;
    std::vector<size_t> reshapeValue;
    std::vector<size_t> permuteValue;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    InferenceEngine::Precision,
    std::string,
    PermuteTransformationTestValues> PermuteTransformationParams;

class PermuteTransformation :
    public testing::WithParamInterface<PermuteTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PermuteTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
