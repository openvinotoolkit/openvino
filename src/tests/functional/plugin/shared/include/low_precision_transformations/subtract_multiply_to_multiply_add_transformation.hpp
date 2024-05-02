// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class SubtractMultiplyToMultiplyAddTransformationTestValues {
public:
    ov::PartialShape inputShape;
    ov::element::Type precision;
    ov::builder::subgraph::FakeQuantizeOnData fqOnData;
};

typedef std::tuple<std::string, SubtractMultiplyToMultiplyAddTransformationTestValues>
    SubtractMultiplyToMultiplyAddTransformationParams;

class SubtractMultiplyToMultiplyAddTransformation
    : public testing::WithParamInterface<SubtractMultiplyToMultiplyAddTransformationParams>,
      public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(
        const testing::TestParamInfo<SubtractMultiplyToMultiplyAddTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
