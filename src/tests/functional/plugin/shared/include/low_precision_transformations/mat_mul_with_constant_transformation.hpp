// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/fake_quantize_on_weights.hpp"
#include "lpt_ov_models/common/constant.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"

#include "lpt_ov_models/mat_mul_function.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class MatMulWithConstantTransformationTestValues {
public:
    ngraph::PartialShape inputShape;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fqOnData;

    ov::builder::subgraph::Constant weights;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fqOnWeights;
    ov::builder::subgraph::DequantizationOperations deqOnWeights;

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
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulWithConstantTransformationParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;

    void Run() override;
};

}  // namespace LayerTestsDefinitions
