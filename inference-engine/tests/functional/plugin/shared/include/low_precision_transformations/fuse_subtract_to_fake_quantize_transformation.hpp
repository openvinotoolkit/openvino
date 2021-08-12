// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/common/add.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class FuseSubtractToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
};

typedef std::tuple<
    std::string,
    FuseSubtractToFakeQuantizeTransformationTestValues> FuseSubtractToFakeQuantizeTransformationParams;

class FuseSubtractToFakeQuantizeTransformation :
    public testing::WithParamInterface<FuseSubtractToFakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseSubtractToFakeQuantizeTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
