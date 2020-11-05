// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class AddTestValues{
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    bool broadcast;
    std::vector<ngraph::element::Type> precisionOnActivations;
    std::vector<ngraph::element::Type> expectedPrecisions;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    AddTestValues
> AddTransformationParams;

class AddTransformation :
    public testing::WithParamInterface<AddTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AddTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
