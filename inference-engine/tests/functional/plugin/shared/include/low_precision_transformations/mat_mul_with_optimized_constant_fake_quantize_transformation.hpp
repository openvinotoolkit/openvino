// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnWeights;
};

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues
> MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams;

class MatMulWithOptimizedConstantFakeQuantizeTransformation :
    public testing::WithParamInterface<MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
