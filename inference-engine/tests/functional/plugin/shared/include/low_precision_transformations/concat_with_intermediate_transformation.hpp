// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string, // target device: CPU, GPU
    InferenceEngine::details::LayerTransformation::Params, // transformation parameters
    bool, // transparent intermediate
    // multichannel
    bool> ConcatWithIntermediateTransformationParams;

class ConcatWithIntermediateTransformation : public LayerTestsUtils::LayerTransformation<ConcatWithIntermediateTransformationParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConcatWithIntermediateTransformationParams> obj);

protected:
    void SetUp() override;
    void validate();
};

}  // namespace LayerTestsDefinitions
