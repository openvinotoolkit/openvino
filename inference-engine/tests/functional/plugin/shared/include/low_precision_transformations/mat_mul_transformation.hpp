// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/mat_mul_function.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class MatMulTransformationTestValues {
public:
    ngraph::Shape inputShape1;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ngraph::Shape inputShape2;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    MatMulTransformationTestValues> MatMulTransformationParams;

class MatMulTransformation :
    public testing::WithParamInterface<MatMulTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
