// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/concat.hpp"

namespace ov {
namespace test {

std::string ConcatLayerTest::getTestCaseName(const testing::TestParamInfo<concatParamsTuple>& obj) {
    int axis;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string targetName;
    std::tie(axis, shapes, model_type, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "axis=" << axis << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ConcatLayerTest::SetUp() {
    int axis;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(axis, shapes, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::ParameterVector params;
    ov::NodeVector params_nodes;
    for (const auto& shape : inputDynamicShapes) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
        params.push_back(param);
        params_nodes.push_back(param);
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(params_nodes, axis);
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    function = std::make_shared<ov::Model>(result, params, "concat");
}

std::string ConcatStringLayerTest::getTestCaseName(const testing::TestParamInfo<ConcatStringParamsTuple>& obj) {
    int axis;
    std::vector<ov::Shape> shapes;
    ov::element::Type model_type;
    std::string targetName;
    std::vector<std::vector<std::string>> data;
    std::tie(axis, shapes, model_type, targetName, data) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << shapes[i] << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")";
    result << "axis=" << axis << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ConcatStringLayerTest::SetUp() {
    int axis;
    std::vector<ov::Shape> shapes;
    ov::element::Type model_type;
    std::tie(axis, shapes, model_type, targetDevice, string_data) = this->GetParam();
    init_input_shapes(ov::test::static_shapes_to_test_representation(shapes));

    ov::ParameterVector params;
    ov::NodeVector params_nodes;
    for (const auto& shape : shapes) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
        params.push_back(param);
        params_nodes.push_back(param);
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(params_nodes, axis);
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    function = std::make_shared<ov::Model>(result, params, "concat");
}

void ConcatStringLayerTest::generate_inputs(const std::vector<ov::Shape>& target_shapes) {
    inputs.clear();
    const auto& model_inputs = function->inputs();
    for (size_t i = 0; i < model_inputs.size(); ++i) {
        inputs.insert({model_inputs[i].get_node_shared_ptr(),
                       ov::Tensor(element::string, model_inputs[i].get_shape(), string_data[i].data())});
    }
}

}  // namespace test
}  // namespace ov
