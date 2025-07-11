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
    const auto& [input_shapes, default_value, values_type, indices_type, in_type, dev] = obj.param;

    for (size_t s = 0; s < input_shapes.size(); ++s) {
        const auto& shape_item = input_shapes[s];
        result << "IS" << s << "=(" << shape_item.first << ")_TS=";
        for (size_t i = 0; i < shape_item.second.size(); ++i) {
            result << "{" << ov::test::utils::vec2str(shape_item.second[i]) << "}_";
        }
    }
    result << "DefaultValue=" << default_value[0] << "_";
    result << "ValuesType=" << values_type << "_";
    result << "IndicesType=" << indices_type << "_";
    result << "InputType=" << in_type << "_";
    result << "Device=" << dev;
    return result.str();
}

void SparseFillEmptyRowsLayerTest::SetUp() {
    const auto& [input_shapes, default_value, values_type, indices_type, in_type, dev] = this->GetParam();
    targetDevice = dev;
    abs_threshold = 1e-5;
    init_input_shapes(input_shapes);

    auto indices = std::make_shared<ov::op::v0::Parameter>(indices_type, inputDynamicShapes[0]);
    auto values = std::make_shared<ov::op::v0::Parameter>(values_type, inputDynamicShapes[1]);
    auto dense_shape = std::make_shared<ov::op::v0::Parameter>(indices_type, inputDynamicShapes[2]);
    std::shared_ptr<ov::Node> default_value_node;
    if (in_type == utils::InputLayerType::PARAMETER) {
        default_value_node = std::make_shared<ov::op::v0::Parameter>(values_type, ov::Shape{});
    } else {
        default_value_node = std::make_shared<ov::op::v0::Constant>(values_type, ov::Shape{}, default_value);
    }

    auto sfe_rows = std::make_shared<ov::op::v16::SparseFillEmptyRows>(indices, values, dense_shape, default_value_node);
    if (in_type == utils::InputLayerType::PARAMETER) {
        function = std::make_shared<ov::Model>(
            sfe_rows->outputs(),
            ov::ParameterVector{indices, values, dense_shape, std::dynamic_pointer_cast<ov::op::v0::Parameter>(default_value_node)});
    } else {
        function = std::make_shared<ov::Model>(sfe_rows->outputs(), ov::ParameterVector{indices, values, dense_shape});
    }
}

const SparseFillEmptyRowsLayerTest::TGenData SparseFillEmptyRowsLayerTest::GetTestDataForDevice(const char* deviceName) {
    const std::vector<std::vector<InputShape>> input_shapes = {{
        // indices: [M, 2], values: [M], dense_shape: [2]
        {{{2, 2}, {{4, 2}}}, {{4}, {{4}}}, {{2}, {{2}}}},
        {{{6, 2}, {{8, 2}}}, {{8}, {{8}}}, {{2}, {{2}}}}
    }};
    const std::vector<std::vector<float>> default_values = {{0.0f}, {42.0f}};
    const std::vector<ov::element::Type> values_types = {ov::element::f32, ov::element::f16};
    const std::vector<ov::element::Type> indices_types = {ov::element::i32, ov::element::i64};
    std::vector<utils::InputLayerType> in_types = {utils::InputLayerType::CONSTANT, utils::InputLayerType::PARAMETER};

    auto data = ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(default_values),
        ::testing::ValuesIn(values_types),
        ::testing::ValuesIn(indices_types),
        ::testing::ValuesIn(in_types),
        ::testing::Values(deviceName));
    return data;
}

}  // namespace test
}  // namespace ov
