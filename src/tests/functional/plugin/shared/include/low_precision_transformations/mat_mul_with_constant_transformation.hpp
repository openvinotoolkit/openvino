// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

#include "ov_lpt_models/mat_mul.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class MatMulWithConstantTransformationTestValues {
public:
    ngraph::PartialShape inputShape;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fqOnData;

    ngraph::builder::subgraph::Constant weights;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fqOnWeights;
    ngraph::builder::subgraph::DequantizationOperations deqOnWeights;

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
