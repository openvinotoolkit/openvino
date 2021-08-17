// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnWeights;
};

typedef std::tuple<
    ngraph::element::Type,
    std::pair<ngraph::PartialShape, ngraph::Shape>,
    std::string,
    MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues
> MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams;

class MatMulWithOptimizedConstantFakeQuantizeTransformation :
    public testing::WithParamInterface<MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
