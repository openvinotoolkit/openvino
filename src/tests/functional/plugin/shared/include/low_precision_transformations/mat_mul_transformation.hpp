// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/mat_mul.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class MatMulTransformationTestValues {
public:
    ngraph::Shape inputShape1;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ngraph::Shape inputShape2;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData2;
    std::string expectedKernelName;
    std::string expectedRuntimePrecision;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    MatMulTransformationTestValues> MatMulTransformationParams;

class MatMulTransformation :
    public testing::WithParamInterface<MatMulTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulTransformationParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
