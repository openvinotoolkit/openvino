// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ov_models/common/constant.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/fake_quantize_on_weights.hpp"
#include "lpt_ov_models/common/reshape.hpp"
#include "lpt_ov_models/common/transpose.hpp"

namespace LayerTestsDefinitions {

class PullReshapeThroughDequantizationTestValues {
public:
    ngraph::element::Type precisionBeforeDequantization;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    ov::builder::subgraph::DequantizationOperations dequantizationOnActivations;
    ov::builder::subgraph::Constant weights;
    ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    ov::builder::subgraph::Reshape reshape1;
    ov::builder::subgraph::DequantizationOperations::Multiply multiply;
    ov::builder::subgraph::Transpose transpose;
    ov::builder::subgraph::Reshape reshape2;
    ngraph::element::Type precisionAfterOperation;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    std::string operationName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ngraph::Shape,
    PullReshapeThroughDequantizationTestValues> PullReshapeThroughDequantizationParams;

class PullReshapeThroughDequantizationTransformation :
    public testing::WithParamInterface<PullReshapeThroughDequantizationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PullReshapeThroughDequantizationParams>& obj);

protected:
    void SetUp() override;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
