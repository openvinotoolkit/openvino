// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_fill_empty_rows.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"
#include "openvino/op/sparse_fill_empty_rows.hpp"

using namespace CPUTestUtils;

namespace ov::test::SparseFillEmptyRows {

std::string SparseFillEmptyRowsLayerCPUTest::getTestCaseName(testing::TestParamInfo<SparseFillEmptyRowsLayerCPUTestParamsSet> obj) {
    const auto& [basicParamsSet, cpuParams] = obj.param;
    const auto& [sparseFillEmptyRowsPar, valuesPrecision, indicesPrecision, secondaryInputType, td] = basicParamsSet;
    const auto& [valuesShape, indicesShape, denseShapeValues, defaultValue] = sparseFillEmptyRowsPar;
    std::ostringstream result;

    result << "valuesShape=" << ov::test::utils::partialShape2str({valuesShape.first}) << "_";
    result << "valuesTS=";
    result << "(";
    for (const auto& targetShape : valuesShape.second) {
        result << ov::test::utils::vec2str(targetShape);
    }
    result << ")";

    result << "_indicesShape=" << ov::test::utils::partialShape2str({indicesShape.first}) << "_";
    result << "indicesTS=";
    result << "(";
    for (const auto& targetShape : indicesShape.second) {
        result << ov::test::utils::vec2str(targetShape);
    }
    result << ")";

    result << "_denseShape=" << ov::test::utils::vec2str(denseShapeValues);
    result << "_defaultValue=" << defaultValue;
    result << "_valuesPrecision=" << valuesPrecision;
    result << "_indicesPrecision=" << indicesPrecision;
    result << "_secondaryInputType=" << secondaryInputType;
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

void SparseFillEmptyRowsLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    const auto secondaryInputType = std::get<3>(std::get<0>(this->GetParam()));

    if (secondaryInputType == ov::test::utils::InputLayerType::CONSTANT) {
        const auto valuesPrecision = funcInputs[0].get_element_type();
        const auto& valuesShape = targetInputStaticShapes[0];
        ov::test::utils::InputGenerateData valuesData;
        valuesData.start_from = 1;
        valuesData.range = 10;
        const auto valuesTensor = ov::test::utils::create_and_fill_tensor(valuesPrecision, valuesShape, valuesData);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), valuesTensor});

        const auto indicesPrecision = funcInputs[1].get_element_type();
        ov::Shape indicesShape = {valuesShape[0], 2};
        ov::test::utils::InputGenerateData indicesData;
        indicesData.start_from = 0;
        indicesData.range = 5;
        const auto indicesTensor = ov::test::utils::create_and_fill_tensor(indicesPrecision, indicesShape, indicesData);
        inputs.insert({funcInputs[1].get_node_shared_ptr(), indicesTensor});

    } else {
        const auto valuesPrecision = funcInputs[0].get_element_type();
        const auto& valuesShape = targetInputStaticShapes[0];
        ov::test::utils::InputGenerateData valuesData;
        valuesData.start_from = 1;
        valuesData.range = 10;
        const auto valuesTensor = ov::test::utils::create_and_fill_tensor(valuesPrecision, valuesShape, valuesData);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), valuesTensor});

        const auto indicesPrecision = funcInputs[2].get_element_type();
        const auto& denseShapeShape = targetInputStaticShapes[1]; // {2}

        const auto& denseShapeValues = std::get<2>(std::get<0>(std::get<0>(this->GetParam())));
        auto denseShapeTensor = ov::Tensor(indicesPrecision, denseShapeShape);
        if (indicesPrecision == ov::element::i32) {
            auto* data_ptr = denseShapeTensor.data<int32_t>();
            for (size_t i = 0; i < denseShapeValues.size(); i++) {
                data_ptr[i] = static_cast<int32_t>(denseShapeValues[i]);
            }
        } else {
            auto* data_ptr = denseShapeTensor.data<int64_t>();
            for (size_t i = 0; i < denseShapeValues.size(); i++) {
                data_ptr[i] = denseShapeValues[i];
            }
        }
        inputs.insert({funcInputs[1].get_node_shared_ptr(), denseShapeTensor});

        ov::Shape indicesShape = {valuesShape[0], 2};
        ov::test::utils::InputGenerateData indicesData;
        indicesData.start_from = 0;
        indicesData.range = 5;
        const auto indicesTensor = ov::test::utils::create_and_fill_tensor(indicesPrecision, indicesShape, indicesData);
        inputs.insert({funcInputs[2].get_node_shared_ptr(), indicesTensor});

        const auto defaultValue = std::get<3>(std::get<0>(std::get<0>(this->GetParam())));
        const auto defaultValueTensor = ov::test::utils::create_and_fill_tensor(valuesPrecision, {}, defaultValue);
        inputs.insert({funcInputs[3].get_node_shared_ptr(), defaultValueTensor});
    }
}

void SparseFillEmptyRowsLayerCPUTest::SetUp() {
    const auto& [basicParamsSet, cpuParams] = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    const auto& [sparseFillEmptyRowsParams, valuesPrecision, indicesPrecision, secondaryInputType, _targetDevice] =
        basicParamsSet;
    targetDevice = _targetDevice;
    const auto& [valuesShape, indicesShape, denseShapeValues, defaultValue] = sparseFillEmptyRowsParams;
    std::vector<ov::test::InputShape> input_shapes = {
        valuesShape,
        {ov::PartialShape{static_cast<ov::Dimension::value_type>(denseShapeValues.size())},
         std::vector<ov::Shape>{ov::Shape{static_cast<size_t>(denseShapeValues.size())}}},
        indicesShape,
        {ov::PartialShape{}, std::vector<ov::Shape>{ov::Shape{}}}
    };
    init_input_shapes(input_shapes);

    auto valuesParameter = std::make_shared<ov::op::v0::Parameter>(valuesPrecision, inputDynamicShapes[0]);
    auto indicesParameter = std::make_shared<ov::op::v0::Parameter>(indicesPrecision, inputDynamicShapes[2]);
    ov::ParameterVector params{ valuesParameter };
    std::shared_ptr<ov::Node> sparseFillEmptyRows;
    if (secondaryInputType == ov::test::utils::InputLayerType::CONSTANT) {
        params.push_back(indicesParameter);
        auto denseShapeConst = std::make_shared<ov::op::v0::Constant>(
            indicesPrecision, ov::Shape{denseShapeValues.size()}, denseShapeValues);
        auto defaultValueConst = std::make_shared<ov::op::v0::Constant>(
            valuesPrecision, ov::Shape{}, defaultValue);
        sparseFillEmptyRows = std::make_shared<ov::op::v16::SparseFillEmptyRows>(
            valuesParameter, denseShapeConst, indicesParameter, defaultValueConst);
    } else {
        auto denseShapeParameter = std::make_shared<ov::op::v0::Parameter>(indicesPrecision, inputDynamicShapes[1]);
        auto defaultValueParameter = std::make_shared<ov::op::v0::Parameter>(valuesPrecision, inputDynamicShapes[3]);
        params.push_back(denseShapeParameter);
        params.push_back(indicesParameter);
        params.push_back(defaultValueParameter);
        sparseFillEmptyRows = std::make_shared<ov::op::v16::SparseFillEmptyRows>(
            valuesParameter, denseShapeParameter, indicesParameter, defaultValueParameter);
    }
    function = makeNgraphFunction(valuesPrecision, params, sparseFillEmptyRows, "SparseFillEmptyRows");
}

TEST_P(SparseFillEmptyRowsLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "SparseFillEmptyRows");
}

const std::vector<ov::test::utils::InputLayerType> secondaryInputTypes = {
    ov::test::utils::InputLayerType::CONSTANT,
    ov::test::utils::InputLayerType::PARAMETER
};

const std::vector<ElementType> indicesPrecisions = {
    ov::element::i32,
    ov::element::i64
};

const std::vector<SparseFillEmptyRowsSpecificParams> SparseFillEmptyRowsParamsVector = {
    // Basic example
    SparseFillEmptyRowsSpecificParams {
        InputShape{{}, {{6}}},                          // values shape
        InputShape{{}, {{6, 2}}},                       // indices shape
        std::vector<int64_t>{10, 10},                   // dense_shape
        1                                               // default_value
    },
    // Dynamic values shape, sequential inference
    SparseFillEmptyRowsSpecificParams {
        InputShape{{-1}, {{3}, {5}, {8}}},              // values shape
        InputShape{{-1, 2}, {{3, 2}, {5, 2}, {8, 2}}},  // indices shape
        std::vector<int64_t>{10, 5},                    // dense_shape
        0                                               // default_value
    },
    // Empty values and indices tensors
    SparseFillEmptyRowsSpecificParams {
        InputShape{{}, {{0}}},                          // values shape
        InputShape{{}, {{0, 2}}},                       // indices shape
        std::vector<int64_t>{5, 5},                     // dense_shape
        99                                              // default_value
    },
    // Different dense_shape
    SparseFillEmptyRowsSpecificParams {
        InputShape{{}, {{10}}},                         // values shape
        InputShape{{}, {{10, 2}}},                      // indices shape
        std::vector<int64_t>{20, 30},                   // dense_shape
        7                                               // default_value
    },
};
}  // namespace ov::test::SparseFillEmptyRows
