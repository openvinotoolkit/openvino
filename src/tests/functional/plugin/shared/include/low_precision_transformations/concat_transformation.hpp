// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class ConcatTransformationTestValues {
public:
    std::shared_ptr<ov::op::v0::Constant> input_constant1;
    ov::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ov::builder::subgraph::DequantizationOperations dequantization1;
    std::shared_ptr<ov::op::v0::Constant> input_constant2;
    ov::builder::subgraph::FakeQuantizeOnData fqOnData2;
    ov::builder::subgraph::DequantizationOperations dequantization2;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ConcatTransformationTestValues> ConcatTransformationParams;

class ConcatTransformation :
    public testing::WithParamInterface<ConcatTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
