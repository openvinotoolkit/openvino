// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

#include "low_precision/recurrent_cell.hpp"

#include "ov_lpt_models/recurrent_cell.hpp"

namespace LayerTestsDefinitions {

class RecurrentCellTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_X;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert_X;
    ngraph::builder::subgraph::DequantizationOperations dequantization_X;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_H;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert_H;
    ngraph::builder::subgraph::DequantizationOperations dequantization_H;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_W;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert_W;
    ngraph::builder::subgraph::DequantizationOperations dequantization_W;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_R;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert_R;
    ngraph::builder::subgraph::DequantizationOperations dequantization_R;
    ngraph::builder::subgraph::RecurrentCellFunction::RNNType RNNType;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    std::vector<ngraph::PartialShape>,
    std::vector<ngraph::Shape>,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    RecurrentCellTransformationParam
>RecurrentCellTransformationParams;

class RecurrentCellTransformation :
    public testing::WithParamInterface<RecurrentCellTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RecurrentCellTransformationParams> obj);

protected:
    void SetUp() override;

    void Run() override;
};

}  // namespace LayerTestsDefinitions
