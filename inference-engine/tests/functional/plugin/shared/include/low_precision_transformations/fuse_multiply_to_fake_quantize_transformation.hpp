// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/common/add.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class FuseMultiplyToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
};

typedef std::tuple<
    std::string,
    FuseMultiplyToFakeQuantizeTransformationTestValues> FuseMultiplyToFakeQuantizeTransformationParams;

class FuseMultiplyToFakeQuantizeTransformation :
    public testing::WithParamInterface<FuseMultiplyToFakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseMultiplyToFakeQuantizeTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
