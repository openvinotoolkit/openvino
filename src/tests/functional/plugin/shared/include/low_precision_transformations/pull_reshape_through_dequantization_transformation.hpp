// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/reshape.hpp"
#include "lpt_ngraph_functions/common/transpose.hpp"

namespace LayerTestsDefinitions {

class PullReshapeThroughDequantizationTestValues {
public:
    ngraph::element::Type precisionBeforeDequantization;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOnActivations;
    ngraph::builder::subgraph::Constant weights;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    ngraph::builder::subgraph::Reshape reshape1;
    ngraph::builder::subgraph::DequantizationOperations::Multiply multiply;
    ngraph::builder::subgraph::Transpose transpose;
    ngraph::builder::subgraph::Reshape reshape2;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    std::string operationName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
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
