// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"

namespace LayerTestsDefinitions {

class GroupConvolutionTransformationParam {
public:
    ngraph::Shape inputShape;
    ngraph::Shape outputShape;
    size_t group;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    GroupConvolutionTransformationParam
> GroupConvolutionTransformationParams;

class GroupConvolutionTransformation :
    public testing::WithParamInterface<GroupConvolutionTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GroupConvolutionTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validateNGraph();
};

}  // namespace LayerTestsDefinitions
