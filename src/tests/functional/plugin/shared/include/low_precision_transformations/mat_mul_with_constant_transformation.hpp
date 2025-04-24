// Copyright (C) 2018-2025 Intel Corporation
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
    ov::PartialShape inputShape;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fqOnData;

    ov::builder::subgraph::Constant weights;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fqOnWeights;
    ov::builder::subgraph::DequantizationOperations deqOnWeights;

    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ov::element::Type,
    std::string,
    MatMulWithConstantTransformationTestValues> MatMulWithConstantTransformationParams;

class MatMulWithConstantTransformation :
    public testing::WithParamInterface<MatMulWithConstantTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulWithConstantTransformationParams>& obj);

protected:
    void SetUp() override;

    void run() override;
};

}  // namespace LayerTestsDefinitions
