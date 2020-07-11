// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

// TODO: refactor: move to LayerTestsUtils::LayerTransformationParams
typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    // TODO: refactor: CNNNetwork LPT is detected
    InferenceEngine::details::LayerTransformation::Params,
    LayerTestsUtils::LayerTransformation::LptVersion> DepthToSpaceTransformationParams;

class DepthToSpaceTransformation :
    public testing::WithParamInterface<DepthToSpaceTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
