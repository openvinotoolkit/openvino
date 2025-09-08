// Copyright (C) 2018-2025 Intel Corporation
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
    ov::Shape inputShape1;
    ov::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ov::Shape inputShape2;
    ov::builder::subgraph::FakeQuantizeOnData fqOnData2;
    std::string expectedKernelName;
    std::string expectedRuntimePrecision;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    MatMulTransformationTestValues> MatMulTransformationParams;

class MatMulTransformation :
    public testing::WithParamInterface<MatMulTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulTransformationParams>& obj);

protected:
    void SetUp() override;
    void run() override;
};

}  // namespace LayerTestsDefinitions
