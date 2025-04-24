// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {
class AssignAndReadValueTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
};

typedef std::tuple <
    ov::element::Type,         // input precision
    ov::PartialShape,          // input shape
    size_t,                        // opset version
    std::string,                   // device
    ov::pass::low_precision::LayerTransformation::Params, // transformation params
    AssignAndReadValueTransformationParam       // test params
> AssignAndReadValueTransformationParams;

class AssignAndReadValueTransformation :
    public testing::WithParamInterface<AssignAndReadValueTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AssignAndReadValueTransformationParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
