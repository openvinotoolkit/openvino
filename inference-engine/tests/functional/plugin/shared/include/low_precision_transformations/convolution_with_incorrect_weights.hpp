// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"

namespace LayerTestsDefinitions {

class ConvolutionWIthIncorrectWeightsParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    bool isCorrect;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ConvolutionWIthIncorrectWeightsParam
> ConvolutionWIthIncorrectWeightsParams;

class ConvolutionWIthIncorrectWeightsTransformation :
    public testing::WithParamInterface<ConvolutionWIthIncorrectWeightsParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionWIthIncorrectWeightsParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
