// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/fake_quantize_on_weights.hpp"

#include "low_precision/recurrent_cell.hpp"

#include "lpt_ov_models/recurrent_cell_function.hpp"

namespace LayerTestsDefinitions {

class RecurrentCellTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_X;
    ov::builder::subgraph::DequantizationOperations::Convert convert_X;
    ov::builder::subgraph::DequantizationOperations dequantization_X;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_H;
    ov::builder::subgraph::DequantizationOperations::Convert convert_H;
    ov::builder::subgraph::DequantizationOperations dequantization_H;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_W;
    ov::builder::subgraph::DequantizationOperations::Convert convert_W;
    ov::builder::subgraph::DequantizationOperations dequantization_W;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_R;
    ov::builder::subgraph::DequantizationOperations::Convert convert_R;
    ov::builder::subgraph::DequantizationOperations dequantization_R;
    ov::builder::subgraph::RecurrentCellFunction::RNNType RNNType;
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
