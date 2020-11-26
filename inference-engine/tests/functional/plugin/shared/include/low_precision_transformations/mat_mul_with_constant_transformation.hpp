// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"
#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class MatMulWithConstantTransformationTestValues {
public:
    ngraph::Shape inputShape;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    ngraph::Shape weightsConstShape;
    std::vector<float> weightsConstValues;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    MatMulWithConstantTransformationTestValues> MatMulWithConstantTransformationParams;

class MatMulWithConstantTransformation :
    public testing::WithParamInterface<MatMulWithConstantTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulWithConstantTransformationParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;

    void Run() override;
};

}  // namespace LayerTestsDefinitions
