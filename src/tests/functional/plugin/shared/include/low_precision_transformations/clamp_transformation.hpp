// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {
class ClampTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    double clampLowConst;
    double clampHighConst;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ClampTransformationParam
> ClampTransformationParams;

class ClampTransformation :
    public testing::WithParamInterface<ClampTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ClampTransformationParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
