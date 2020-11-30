// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"


namespace LayerTestsDefinitions {
class ReshapeTransformationParam {
public:
    ngraph::Shape inputShape;
    std::vector<int> reshapeConstValues;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
};

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ReshapeTransformationParam
> ReshapeTransformationParams;

class ReshapeTransformation :
    public testing::WithParamInterface<ReshapeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReshapeTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
