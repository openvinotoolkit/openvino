// Copyright (C) 2020 Intel Corporation
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
    std::string,
    InferenceEngine::details::LayerTransformation::Params,
    LayerTestsUtils::LayerTransformation::LptVersion,
    bool,
    bool> NormalizeTransformationParams;

class NormalizeTransformation :
    public testing::WithParamInterface<NormalizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NormalizeTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
