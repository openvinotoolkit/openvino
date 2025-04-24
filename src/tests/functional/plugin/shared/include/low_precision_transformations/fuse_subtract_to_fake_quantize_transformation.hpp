// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ov_lpt_models/common/add.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class FuseSubtractToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
};

typedef std::tuple<
    std::string,
    FuseSubtractToFakeQuantizeTransformationTestValues> FuseSubtractToFakeQuantizeTransformationParams;

class FuseSubtractToFakeQuantizeTransformation :
    public testing::WithParamInterface<FuseSubtractToFakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseSubtractToFakeQuantizeTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
