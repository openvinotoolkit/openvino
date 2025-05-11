// Copyright (C) 2018-2025 Intel Corporation
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
    ov::element::Type,
    std::vector<ov::PartialShape>,
    std::vector<ov::Shape>,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    bool, // use precision transparent operations
    RecurrentCellTransformationParam
>RecurrentCellTransformationParams;

class RecurrentCellTransformation :
    public testing::WithParamInterface<RecurrentCellTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RecurrentCellTransformationParams> obj);

protected:
    void SetUp() override;

    void run() override;
};

}  // namespace LayerTestsDefinitions
