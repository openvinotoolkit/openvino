// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

using namespace ngraph;

namespace LayerTestsDefinitions {

typedef std::tuple <
    element::Type,
    Shape,
    std::string,
    builder::subgraph::FakeQuantizeOnData> MultiplyToGroupConvolutionTransformationParams;

class MultiplyToGroupConvolutionTransformation :
    public testing::WithParamInterface<MultiplyToGroupConvolutionTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MultiplyToGroupConvolutionTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
