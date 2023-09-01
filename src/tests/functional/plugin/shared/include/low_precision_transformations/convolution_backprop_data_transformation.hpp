// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <utility>


#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/fake_quantize_on_weights.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {

class ConvolutionBackpropDataTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    std::string layerName;
    std::string expectedKernelType;

    ConvolutionBackpropDataTransformationParam() = default;
    ConvolutionBackpropDataTransformationParam(
        const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
        const ov::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        std::string layerName,
        std::string expectedKernelType) :
        fakeQuantizeOnData(fakeQuantizeOnData), fakeQuantizeOnWeights(fakeQuantizeOnWeights),
        layerName(std::move(layerName)), expectedKernelType(std::move(expectedKernelType)) {}
    ConvolutionBackpropDataTransformationParam(
        const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
        ov::builder::subgraph::DequantizationOperations  dequantizationOnWeights,
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
