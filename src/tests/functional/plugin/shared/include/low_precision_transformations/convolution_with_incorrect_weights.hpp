// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/fake_quantize_on_weights.hpp"

namespace LayerTestsDefinitions {

class ConvolutionWIthIncorrectWeightsParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    bool isCorrect;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ConvolutionWIthIncorrectWeightsParam
> ConvolutionWIthIncorrectWeightsParams;

class ConvolutionWIthIncorrectWeightsTransformation :
    public testing::WithParamInterface<ConvolutionWIthIncorrectWeightsParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvolutionWIthIncorrectWeightsParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
