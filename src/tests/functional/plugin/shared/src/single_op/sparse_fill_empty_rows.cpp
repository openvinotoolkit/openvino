// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/sparse_fill_empty_rows.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {
std::string SparseFillEmptyRowsLayerTest::getTestCaseName(const testing::TestParamInfo<SparseFillEmptyRowsParams>& obj) {
    std::ostringstream result;
    const auto& [input_shapes, default_value, values_type, in_type, dev] = obj.param;

    for (size_t s = 0; s < input_shapes.size(); ++s) {
        const auto& shape_item = input_shapes[s];
        result << "IS" << s << "=(" << shape_item.first << ")_TS=";
        for (size_t i = 0; i < shape_item.second.size(); ++i) {
            result << "{" << ov::test::utils::vec2str(shape_item.second[i]) << "}_";
        }
    }
    result << "DefaultValue=" << default_value[0] << "_";
    result << "ValuesType=" << values_type << "_";
    result << "InputType=" << in_type << "_";
    result << "Device=" << dev;
    return result.str();
}

void SparseFillEmptyRowsLayerTest::SetUp() {
    const auto& [input_shapes, default_value, values_type, in_type, dev] = this->GetParam();
    targetDevice = dev;
    abs_threshold = 2e-5;
    init_input_shapes(input_shapes);

    auto values = std::make_shared<ov::op::v0::Parameter>(values_type, inputDynamicShapes[0]);
    auto dense_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]);
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[2]);
    std::shared_ptr<ov::Node> default_value_node;
    if (in_type == utils::InputLayerType::PARAMETER) {
        default_value_node = std::make_shared<ov::op::v0::Parameter>(values_type, ov::PartialShape{});
    } else {
        default_value_node = std::make_shared<ov::op::v0::Constant>(values_type, ov::Shape{}, default_value);
    }

    auto sfe_rows = std::make_shared<ov::op::v16::SparseFillEmptyRows>(values, dense_shape, indices, default_value_node);
    if (in_type == utils::InputLayerType::PARAMETER) {
        function = std::make_shared<ov::Model>(
            sfe_rows->outputs(),
            ov::ParameterVector{values, dense_shape, indices, std::dynamic_pointer_cast<ov::op::v0::Parameter>(default_value_node)});
    } else {
        function = std::make_shared<ov::Model>(sfe_rows->outputs(), ov::ParameterVector{values, dense_shape, indices});
    }
}

void SparseFillEmptyRowsLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    const auto& valuesShape = targetInputStaticShapes[0];
    const auto& denseShapeShape = targetInputStaticShapes[1];
    const auto& indicesShape = targetInputStaticShapes[2];
    const auto valuesType = funcInputs[0].get_element_type();
    const auto idxType = funcInputs[2].get_element_type();

    size_t M = indicesShape[0];
    size_t num_rows = std::max(static_cast<size_t>(8), M + 2);
    size_t num_cols = std::max(static_cast<size_t>(5), static_cast<size_t>(3));

    ov::Tensor dense_shape_tensor(idxType, denseShapeShape);
    auto* ptr = dense_shape_tensor.data<int64_t>();
    ptr[0] = static_cast<int64_t>(num_rows);
    ptr[1] = static_cast<int64_t>(num_cols);
    inputs[funcInputs[1].get_node_shared_ptr()] = dense_shape_tensor;

    ov::test::utils::InputGenerateData valuesData;
    valuesData.start_from = 1;
    valuesData.range = 10;
    auto values_tensor = ov::test::utils::create_and_fill_tensor(valuesType, valuesShape, valuesData);
    inputs[funcInputs[0].get_node_shared_ptr()] = values_tensor;

    // Generate indices: [M, 2], each row in [0, num_rows-1], col in [0, num_cols-1]
    ov::Tensor indices_tensor(idxType, indicesShape);
    auto* iptr = indices_tensor.data<int64_t>();
    for (size_t i = 0; i < M; ++i) {
        iptr[i * 2 + 0] = static_cast<int64_t>(i % num_rows);
        iptr[i * 2 + 1] = static_cast<int64_t>((i * 2) % num_cols);
    }
    inputs[funcInputs[2].get_node_shared_ptr()] = indices_tensor;

    const auto& default_value_type = funcInputs.size() > 3 ? funcInputs[3].get_element_type() : valuesType;
    ov::test::utils::InputGenerateData defvalData;
    defvalData.start_from = 42;
    defvalData.range = 1;
    auto default_value_tensor = ov::test::utils::create_and_fill_tensor(default_value_type, {}, defvalData);
    if (funcInputs.size() > 3)
        inputs[funcInputs[3].get_node_shared_ptr()] = default_value_tensor;
}

const SparseFillEmptyRowsLayerTest::TGenData SparseFillEmptyRowsLayerTest::GetTestDataForDevice(const char* deviceName) {
    const std::vector<std::vector<InputShape>> input_shapes = {{
        // values: [M], dense_shape: [2], indices: [M, 2]
        {{{4}, {{4}}}, {{2}, {{2}}}, {{4, 2}, {{4, 2}}}},
        {{{8}, {{8}}}, {{2}, {{2}}}, {{8, 2}, {{8, 2}}}}
    }};
    const std::vector<std::vector<float>> default_values = {{0.0f}, {42.0f}};
    const std::vector<ov::element::Type> values_types = {ov::element::f32, ov::element::f16};
    std::vector<utils::InputLayerType> in_types = {utils::InputLayerType::CONSTANT, utils::InputLayerType::PARAMETER};

    auto data = ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(default_values),
        ::testing::ValuesIn(values_types),
        ::testing::ValuesIn(in_types),
        ::testing::Values(deviceName));
    return data;
}

const SparseFillEmptyRowsLayerTest::TGenData SparseFillEmptyRowsLayerTest::GetStaticTestDataForDevice(const char* deviceName) {
    const std::vector<std::vector<InputShape>> input_shapes_static = {{
        // values: [M], dense_shape: [2], indices: [M, 2]
        {{{4}, {{4}}}, {{2}, {{2}}}, {{4, 2}, {{4, 2}}}},
        {{{8}, {{8}}}, {{2}, {{2}}}, {{8, 2}, {{8, 2}}}},
        {{{6}, {{6}}}, {{2}, {{2}}}, {{6, 2}, {{6, 2}}}},
        {{{12}, {{12}}}, {{2}, {{2}}}, {{12, 2}, {{12, 2}}}}
    }};

    const std::vector<std::vector<float>> default_values = {{0.0f}, {42.0f}};
    const std::vector<ov::element::Type> values_types = {ov::element::f32, ov::element::f16};
    std::vector<utils::InputLayerType> in_types = {utils::InputLayerType::CONSTANT, utils::InputLayerType::PARAMETER};

    auto data = ::testing::Combine(
        ::testing::ValuesIn(input_shapes_static),
        ::testing::ValuesIn(default_values),
        ::testing::ValuesIn(values_types),
        ::testing::ValuesIn(in_types),
        ::testing::Values(deviceName));
    return data;
}

const SparseFillEmptyRowsLayerTest::TGenData SparseFillEmptyRowsLayerTest::GetDynamicTestDataForDevice(const char* deviceName) {
    const std::vector<std::vector<InputShape>> input_shapes_dynamic = {{
        // values: [M], dense_shape: [2], indices: [M, 2] - with dynamic M dimension
        {{{ov::Dimension(1, 20)}, {{4}, {8}, {6}}}, {{2}, {{2}}}, {{ov::Dimension(1, 20), 2}, {{4, 2}, {8, 2}, {6, 2}}}},
        {{{ov::Dimension(2, 16)}, {{10}, {12}, {14}}}, {{2}, {{2}}}, {{ov::Dimension(2, 16), 2}, {{10, 2}, {12, 2}, {14, 2}}}},
        {{{ov::Dimension(1, 50)}, {{3}, {5}, {7}, {9}}}, {{2}, {{2}}}, {{ov::Dimension(1, 50), 2}, {{3, 2}, {5, 2}, {7, 2}, {9, 2}}}}
    }};

    const std::vector<std::vector<float>> default_values = {{0.0f}, {42.0f}};
    const std::vector<ov::element::Type> values_types = {ov::element::f32, ov::element::f16};
    std::vector<utils::InputLayerType> in_types = {utils::InputLayerType::CONSTANT, utils::InputLayerType::PARAMETER};

    auto data = ::testing::Combine(
        ::testing::ValuesIn(input_shapes_dynamic),
        ::testing::ValuesIn(default_values),
        ::testing::ValuesIn(values_types),
        ::testing::ValuesIn(in_types),
        ::testing::Values(deviceName));
    return data;
}

}  // namespace test
}  // namespace ov
