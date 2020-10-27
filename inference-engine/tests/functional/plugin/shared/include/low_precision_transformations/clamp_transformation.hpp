// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {
class ClampTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    double clampLowConst;
    double clampHighConst;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ClampTransformationParam
> ClampTransformationParams;

class ClampTransformation :
    public testing::WithParamInterface<ClampTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ClampTransformationParams> obj);
protected:
    void SetUp() override;
private:
    void validateNGraph();
};
}  // namespace LayerTestsDefinitions
