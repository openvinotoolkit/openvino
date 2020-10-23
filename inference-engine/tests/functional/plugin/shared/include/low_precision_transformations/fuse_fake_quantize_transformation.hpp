// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/add.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class FuseFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeAdd;
        ngraph::builder::subgraph::Add add;
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type precisionAfterDequantization;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
};

typedef std::tuple<
    std::string,
    FuseFakeQuantizeTransformationTestValues> FuseFakeQuantizeTransformationParams;

class FuseFakeQuantizeTransformation :
    public testing::WithParamInterface<FuseFakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
