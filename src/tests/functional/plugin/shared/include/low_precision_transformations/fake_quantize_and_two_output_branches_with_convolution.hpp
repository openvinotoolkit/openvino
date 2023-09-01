// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/fake_quantize_on_weights.hpp"
#include "lpt_ov_models/fake_quantize_and_two_output_branches_with_convolution_function.hpp"

namespace LayerTestsDefinitions {
class FakeQuantizeAndTwoOutputBranchesWithConvolution {
public:
    ov::builder::subgraph::FakeQuantizeOnData fqOnData;
    ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights1;
    ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    FakeQuantizeAndTwoOutputBranchesWithConvolution
> FakeQuantizeAndTwoOutputBranchesWithConvolutionParams;

class FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation :
    public testing::WithParamInterface<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
