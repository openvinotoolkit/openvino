// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <utility>


#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {

class ConvolutionBackpropDataTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    std::string layerName;
    std::string expectedKernelType;

    ConvolutionBackpropDataTransformationParam() = default;
    ConvolutionBackpropDataTransformationParam(
        const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        std::string layerName,
        std::string expectedKernelType) :
        fakeQuantizeOnData(fakeQuantizeOnData), fakeQuantizeOnWeights(fakeQuantizeOnWeights),
        layerName(std::move(layerName)), expectedKernelType(std::move(expectedKernelType)) {}
    ConvolutionBackpropDataTransformationParam(
        const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
        ngraph::builder::subgraph::DequantizationOperations  dequantizationOnWeights,
        std::string layerName,
        std::string expectedKernelType) :
        fakeQuantizeOnData(fakeQuantizeOnData), dequantizationOnWeights(std::move(dequantizationOnWeights)),
        layerName(std::move(layerName)), expectedKernelType(std::move(expectedKernelType)) {}
};

typedef std::tuple<
    ngraph::element::Type, // netPrecision
    std::pair<ngraph::PartialShape, bool>, // input shape and shape support flag
    ngraph::Shape,         // outputShape
    std::string,           // targetDevice
    ov::pass::low_precision::LayerTransformation::Params,
    ConvolutionBackpropDataTransformationParam
> ConvolutionBackpropDataTransformationParams;

class ConvolutionBackpropDataTransformation :
    public testing::WithParamInterface<ConvolutionBackpropDataTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvolutionBackpropDataTransformationParams>& obj);

protected:
    void SetUp() override;

    void Run() override;
};

}  // namespace LayerTestsDefinitions
