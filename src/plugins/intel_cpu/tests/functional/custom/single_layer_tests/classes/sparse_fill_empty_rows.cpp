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

namespace ov {
namespace test {
namespace SparseFillEmptyRows {

std::string SparseFillEmptyRowsLayerCPUTest::getTestCaseName(testing::TestParamInfo<SparseFillEmptyRowsLayerCPUTestParamsSet> obj) {
    SparseFillEmptyRowsLayerTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    SparseFillEmptyRowsSpecificParams sparseFillEmptyRowsPar;
    std::tie(basicParamsSet, cpuParams) = obj.param;
    std::string td;
    ElementType valuesPrecision;
    ElementType indicesPrecision;
    ov::test::utils::InputLayerType secondaryInputType;
    std::tie(sparseFillEmptyRowsPar, valuesPrecision, indicesPrecision, secondaryInputType, td) = basicParamsSet;

    InputShape valuesShape;
    InputShape indicesShape;
    std::vector<int64_t> denseShapeValues;
    int64_t defaultValue;
    std::tie(valuesShape, indicesShape, denseShapeValues, defaultValue) = sparseFillEmptyRowsPar;
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
    const auto valuesPrecision = funcInputs[0].get_element_type();

    // Values input
    const auto& valuesShape = targetInputStaticShapes[0];
    ov::test::utils::InputGenerateData valuesData;
    valuesData.start_from = 1;
    valuesData.range = 10;
    const auto valuesTensor = ov::test::utils::create_and_fill_tensor(valuesPrecision, valuesShape, valuesData);
    inputs.insert({funcInputs[0].get_node_shared_ptr(), valuesTensor});

    const auto secondaryInputType = std::get<3>(std::get<0>(this->GetParam()));
    if (secondaryInputType == ov::test::utils::InputLayerType::PARAMETER) {
        const auto& denseShapeShape = targetInputStaticShapes[1];
        ov::test::utils::InputGenerateData denseShapeData;
        denseShapeData.start_from = 2;
        denseShapeData.range = 10;
        const auto indicesPrecision = funcInputs[2].get_element_type();
        const auto denseShapeTensor = ov::test::utils::create_and_fill_tensor(indicesPrecision, denseShapeShape, denseShapeData);
        inputs.insert({funcInputs[1].get_node_shared_ptr(), denseShapeTensor});

        const auto& indicesShape = targetInputStaticShapes[2];
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
    SparseFillEmptyRowsLayerTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    std::tie(basicParamsSet, cpuParams) = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    SparseFillEmptyRowsSpecificParams sparseFillEmptyRowsParams;
    ElementType valuesPrecision;
    ElementType indicesPrecision;
    ov::test::utils::InputLayerType secondaryInputType;
    std::tie(sparseFillEmptyRowsParams, valuesPrecision, indicesPrecision, secondaryInputType, targetDevice) = basicParamsSet;

    InputShape valuesShape;
    InputShape indicesShape;
    std::vector<int64_t> denseShapeValues;
    int64_t defaultValue;
    std::tie(valuesShape, indicesShape, denseShapeValues, defaultValue) = sparseFillEmptyRowsParams;

    // Define input shapes
    std::vector<ov::test::InputShape> input_shapes = {
        valuesShape,  // values
        {ov::PartialShape{static_cast<ov::Dimension::value_type>(denseShapeValues.size())},
         std::vector<ov::Shape>{ov::Shape{static_cast<size_t>(denseShapeValues.size())}}}, // dense_shape
        indicesShape, // indices
        {ov::PartialShape{}, std::vector<ov::Shape>{ov::Shape{}}}  // default_value
    };

    init_input_shapes(input_shapes);

    // Create parameters and constants for the operation
    auto valuesParameter = std::make_shared<ov::op::v0::Parameter>(valuesPrecision, inputDynamicShapes[0]);
    ov::ParameterVector params{ valuesParameter };

    std::shared_ptr<ov::Node> sparseFillEmptyRows;

    if (secondaryInputType == ov::test::utils::InputLayerType::CONSTANT) {
        // Create operation with constant inputs
        auto denseShapeConst = std::make_shared<ov::op::v0::Constant>(indicesPrecision, ov::Shape{denseShapeValues.size()}, denseShapeValues);
        auto indicesConst = std::make_shared<ov::op::v0::Constant>(indicesPrecision,
                                                                    ov::Shape{indicesShape.second[0][0], 2},
                                                                    std::vector<int64_t>(indicesShape.second[0][0] * 2, 0));
        auto defaultValueConst = std::make_shared<ov::op::v0::Constant>(valuesPrecision, ov::Shape{}, defaultValue);

        sparseFillEmptyRows = std::make_shared<ov::op::v16::SparseFillEmptyRows>(
            valuesParameter, denseShapeConst, indicesConst, defaultValueConst);
    } else {
        // Create operation with parameter inputs
        auto denseShapeParameter = std::make_shared<ov::op::v0::Parameter>(indicesPrecision, inputDynamicShapes[1]);
        auto indicesParameter = std::make_shared<ov::op::v0::Parameter>(indicesPrecision, inputDynamicShapes[2]);
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
    //SparseFillEmptyRowsSpecificParams {
    //    InputShape{{}, {{6}}},                    // values shape
    //    InputShape{{}, {{6, 2}}},                 // indices shape
    //    std::vector<int64_t>{10, 10},             // dense_shape
    //    1                                         // default_value
    //},
    // Dynamic values shape
    SparseFillEmptyRowsSpecificParams {
        InputShape{{-1}, {{3}, {5}, {8}}},        // values shape
        InputShape{{-1, 2}, {{3, 2}, {5, 2}, {8, 2}}}, // indices shape
        std::vector<int64_t>{10, 5},              // dense_shape
        0                                          // default_value
    },
    // Empty values tensor
    //SparseFillEmptyRowsSpecificParams {
    //    InputShape{{}, {{0}}},                    // values shape
    //    InputShape{{}, {{0, 2}}},                 // indices shape
    //    std::vector<int64_t>{5, 5},               // dense_shape
    //    99                                        // default_value
    //},
    //// Different dense_shape
    //SparseFillEmptyRowsSpecificParams {
    //    InputShape{{}, {{10}}},                   // values shape
    //    InputShape{{}, {{10, 2}}},                // indices shape
    //    std::vector<int64_t>{20, 30},             // dense_shape
    //    7                                          // default_value
    //},
};

}  // namespace SparseFillEmptyRows
}  // namespace test
}  // namespace ov
