// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ngraph/ngraph.hpp>
#include "ov_lpt_models/common/add.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class FuseMultiplyToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
};

typedef std::tuple<
    std::string,
    FuseMultiplyToFakeQuantizeTransformationTestValues> FuseMultiplyToFakeQuantizeTransformationParams;

class FuseMultiplyToFakeQuantizeTransformation :
    public testing::WithParamInterface<FuseMultiplyToFakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseMultiplyToFakeQuantizeTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
