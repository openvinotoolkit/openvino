// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fqOnData;
    ov::builder::subgraph::FakeQuantizeOnData fqOnWeights;
};

typedef std::tuple<
    ov::element::Type,
    std::pair<ov::PartialShape, ov::Shape>,
    std::string,
    MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues
> MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams;

class MatMulWithOptimizedConstantFq :
    public testing::WithParamInterface<MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
