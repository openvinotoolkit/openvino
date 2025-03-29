// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ov_lpt_models/common/add.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class FuseDequantizeToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeAdd;
        ov::builder::subgraph::Add add;
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
        ov::element::Type precisionAfterDequantization;
        ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    };

    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
};

typedef std::tuple<
    std::string,
    FuseDequantizeToFakeQuantizeTransformationTestValues> FuseDequantizeToFakeQuantizeTransformationParams;

class FuseDequantizeToFakeQuantizeTransformation
    : public testing::WithParamInterface<FuseDequantizeToFakeQuantizeTransformationParams>,
      public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseDequantizeToFakeQuantizeTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
