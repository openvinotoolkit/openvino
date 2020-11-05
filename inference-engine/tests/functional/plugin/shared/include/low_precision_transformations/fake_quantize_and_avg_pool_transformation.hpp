// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ngraph::builder::subgraph::FakeQuantizeOnData> FakeQuantizeAndAvgPoolTransformationParams;

class FakeQuantizeAndAvgPoolTransformation :
    public testing::WithParamInterface<FakeQuantizeAndAvgPoolTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeAndAvgPoolTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
