// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class ClampTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    double clampLowConst;
    double clampHighConst;
};

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    InferenceEngine::details::LayerTransformation::Params,
    LayerTestsUtils::LayerTransformation::LptVersion,
    ClampTransformationParam
> ClampTransformationParams;

class ClampTransformation :
    public testing::WithParamInterface<ClampTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ClampTransformationParams> obj);
protected:
    void SetUp() override;
private:
    void validate();
};
}  // namespace LayerTestsDefinitions
