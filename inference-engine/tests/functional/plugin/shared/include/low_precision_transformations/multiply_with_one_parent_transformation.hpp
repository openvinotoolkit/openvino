// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class MultiplyWithOneParentTransformationValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
};

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    MultiplyWithOneParentTransformationValues
> MultiplyWithOneParentTransformationParams;

class MultiplyWithOneParentTransformation :
    public testing::WithParamInterface<MultiplyWithOneParentTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MultiplyWithOneParentTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
