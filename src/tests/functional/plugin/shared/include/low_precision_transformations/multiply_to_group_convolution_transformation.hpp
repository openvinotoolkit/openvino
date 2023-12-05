// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

using namespace ngraph;

namespace LayerTestsDefinitions {

class MultiplyToGroupConvolutionTransformationParam {
public:
    builder::subgraph::FakeQuantizeOnData fqOnData;
    builder::subgraph::Constant constant;
    std::string layerName;
    std::string expectedKernelType;
    bool parentHasOneConsumer;
};

typedef std::tuple <
    element::Type,
    PartialShape,
    std::string,
    MultiplyToGroupConvolutionTransformationParam> MultiplyToGroupConvolutionTransformationParams;

class MultiplyToGroupConvolutionTransformation :
    public testing::WithParamInterface<MultiplyToGroupConvolutionTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyToGroupConvolutionTransformationParams>& obj);

protected:
    void SetUp() override;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
