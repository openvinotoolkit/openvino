// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {

class SubtractMultiplyToMultiplyAddTransformationTestValues {
public:
    ngraph::PartialShape inputShape;
    ngraph::element::Type precision;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
};

typedef std::tuple<
    std::string,
    SubtractMultiplyToMultiplyAddTransformationTestValues> SubtractMultiplyToMultiplyAddTransformationParams;

class SubtractMultiplyToMultiplyAddTransformation :
    public testing::WithParamInterface<SubtractMultiplyToMultiplyAddTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SubtractMultiplyToMultiplyAddTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
