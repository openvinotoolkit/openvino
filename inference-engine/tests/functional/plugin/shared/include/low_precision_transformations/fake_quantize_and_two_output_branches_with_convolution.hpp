// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/fake_quantize_and_two_output_branches_with_convolution_function.hpp"

namespace LayerTestsDefinitions {
class FakeQuantizeAndTwoOutputBranchesWithConvolution {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights1;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    FakeQuantizeAndTwoOutputBranchesWithConvolution
> FakeQuantizeAndTwoOutputBranchesWithConvolutionParams;

class FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation :
    public testing::WithParamInterface<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
