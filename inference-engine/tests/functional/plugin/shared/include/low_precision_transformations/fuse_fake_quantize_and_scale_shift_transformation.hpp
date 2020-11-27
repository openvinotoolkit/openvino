// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "lpt_ngraph_functions/fuse_fake_quantize_and_scale_shift_function.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ngraph::builder::subgraph::FakeQuantizeOnData> FuseFakeQuantizeAndScaleShiftTransformationParams;

class FuseFakeQuantizeAndScaleShiftTransformation :
    public testing::WithParamInterface<FuseFakeQuantizeAndScaleShiftTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeAndScaleShiftTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
