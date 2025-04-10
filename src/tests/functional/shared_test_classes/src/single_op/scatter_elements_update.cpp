// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/scatter_elements_update.hpp"

namespace ov {
namespace test {
std::string ScatterElementsUpdateLayerTest::getTestCaseName(const testing::TestParamInfo<scatterElementsUpdateParamsTuple> &obj) {
    auto shapes_ss = [](const InputShape& shape) {
        std::stringstream ss;
        ss << "_IS=(" << ov::test::utils::partialShape2str({shape.first}) << ")_TS=";
        for (size_t j = 0lu; j < shape.second.size(); j++)
            ss << "{" << ov::test::utils::vec2str(shape.second[j]) << "}";
        return ss;
    };

    axisShapeInShape shapes_desc;
    std::vector<InputShape> input_shapes;
    int axis;
    std::vector<size_t> indices_value;
    ov::element::Type model_type, indices_type;
    std::string target_device;
    std::tie(shapes_desc, indices_value, model_type, indices_type, target_device) = obj.param;
    std::tie(input_shapes, axis) = shapes_desc;
    std::ostringstream result;
    result << "InputShape=" << shapes_ss(input_shapes.at(0)).str() << "_";
    result << "IndicesShape=" << ov::test::utils::vec2str(input_shapes.at(1).second) << "_";
    result << "Axis=" << axis << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "idxType=" << indices_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ScatterElementsUpdateLayerTest::SetUp() {
    axisShapeInShape shapes_desc;
    std::vector<InputShape> input_shapes;
    int axis;
    std::vector<size_t> indices_value;
    ov::element::Type model_type, indices_type;
    std::string target_device;
    std::tie(shapes_desc, indices_value, model_type, indices_type, targetDevice) = this->GetParam();
    std::tie(input_shapes, axis) = shapes_desc;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(0));
    auto update_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(1));
    auto indices_const = std::make_shared<ov::op::v0::Constant>(indices_type, targetStaticShapes.at(0).at(1), indices_value);
    auto axis_const =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int>{axis});
    auto scatter_elements_update = std::make_shared<ov::op::v3::ScatterElementsUpdate>(param, indices_const, update_param, axis_const);
    function = std::make_shared<ov::Model>(scatter_elements_update->outputs(), ov::ParameterVector{param, update_param}, "ScatterElementsUpdate");
}

std::string ScatterElementsUpdate12LayerTest::getTestCaseName(const testing::TestParamInfo<scatterElementsUpdate12ParamsTuple> &obj) {
    auto shapes_ss = [](const InputShape& shape) {
        std::stringstream ss;
        ss << "_IS=(" << ov::test::utils::partialShape2str({shape.first}) << ")_TS=";
        for (size_t j = 0lu; j < shape.second.size(); j++)
            ss << "{" << ov::test::utils::vec2str(shape.second[j]) << "}";
        return ss;
    };

    axisShapeInShape shapes_desc;
    std::vector<InputShape> input_shapes;
    int axis;
    std::vector<int64_t> indices_value;
    ov::op::v12::ScatterElementsUpdate::Reduction reduceMode;
    bool useInitVal;
    ov::element::Type model_type, indices_type;
    std::string target_device;
    std::tie(shapes_desc, indices_value, reduceMode, useInitVal, model_type, indices_type, target_device) = obj.param;
    std::tie(input_shapes, axis) = shapes_desc;
    std::ostringstream result;
    result << "InputShape=" << shapes_ss(input_shapes.at(0)).str() << "_";
    result << "IndicesShape=" << ov::test::utils::vec2str(input_shapes.at(1).second) << "_";
    result << "Axis=" << axis << "_";
    result << "ReduceMode=" << as_string(reduceMode) << "_";
    result << "UseInitVal=" << useInitVal << "_";
    result << "Indices=" << ov::test::utils::vec2str(indices_value) << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "idxType=" << indices_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ScatterElementsUpdate12LayerTest::SetUp() {
    axisShapeInShape shapes_desc;
    std::vector<InputShape> input_shapes;
    int axis;
    std::vector<int64_t> indices_value;
    ov::op::v12::ScatterElementsUpdate::Reduction reduceMode;
    bool useInitVal;
    ov::element::Type model_type, indices_type;
    std::string target_device;
    std::tie(shapes_desc, indices_value, reduceMode, useInitVal, model_type, indices_type, targetDevice) = this->GetParam();
    std::tie(input_shapes, axis) = shapes_desc;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(0));
    auto update_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(1));
    auto indices_const = std::make_shared<ov::op::v0::Constant>(indices_type, targetStaticShapes.at(0).at(1), indices_value);
    auto axis_const =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int>{axis});
    auto scatter_elements_update = std::make_shared<ov::op::v12::ScatterElementsUpdate>(param, indices_const, update_param, axis_const, reduceMode, useInitVal);
    function = std::make_shared<ov::Model>(scatter_elements_update->outputs(), ov::ParameterVector{param, update_param}, "ScatterElementsUpdate");
}
}  // namespace test
}  // namespace ov
