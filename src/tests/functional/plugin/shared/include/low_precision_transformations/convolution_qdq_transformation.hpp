// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

namespace LayerTestsDefinitions {

class ConvolutionQDqTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    ov::builder::subgraph::DequantizationOperations::Convert convertOnData;
    ov::builder::subgraph::DequantizationOperations dequantizationOnData;

    ov::builder::subgraph::Constant constantOnWeights;
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    ov::builder::subgraph::DequantizationOperations::Convert convertOnWeights;
    ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;

    std::string layerName;
    std::string expectedKernelType;
};

inline std::ostream& operator<<(std::ostream& out, const ConvolutionQDqTransformationParam& data) {
    return out <<  "_" <<
        data.fakeQuantizeOnData << "_" <<
        data.convertOnData << "_" <<
        data.dequantizationOnData << "_" <<

        data.constantOnWeights << "_" <<
        data.fakeQuantizeOnWeights << "_" <<
        data.convertOnWeights << "_" <<
        data.dequantizationOnWeights <<

        data.layerName << "_" <<
        data.expectedKernelType;
}

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ConvolutionQDqTransformationParam
> ConvolutionQDqTransformationParams;

class ConvolutionQDqTransformation :
    public testing::WithParamInterface<ConvolutionQDqTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvolutionQDqTransformationParams>& obj);

protected:
    void SetUp() override;

    void run() override;
};

}  // namespace LayerTestsDefinitions
