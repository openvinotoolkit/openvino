// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ngraph/ngraph.hpp>
#include "lpt_ov_models/common/add.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class FuseDequantizeToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeAdd;
        ov::builder::subgraph::Add add;
        ngraph::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type precisionAfterDequantization;
        ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    };

    ngraph::PartialShape inputShape;
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
