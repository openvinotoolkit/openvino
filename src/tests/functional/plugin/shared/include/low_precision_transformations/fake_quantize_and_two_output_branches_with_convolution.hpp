// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/fake_quantize_and_two_output_branches_with_convolution.hpp"

namespace LayerTestsDefinitions {
class FakeQuantizeAndTwoOutputBranchesWithConvolution {
public:
    ov::builder::subgraph::FakeQuantizeOnData fqOnData;
    ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights1;
    ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights2;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
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
