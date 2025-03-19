// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/scatter_update.hpp"

namespace ov {
namespace test {
std::string ScatterUpdateLayerTest::getTestCaseName(const testing::TestParamInfo<scatterUpdateParamsTuple> &obj) {
    auto shapes_ss = [](const InputShape& shape) {
        std::stringstream ss;
        ss << "_IS=(" << ov::test::utils::partialShape2str({shape.first}) << ")_TS=";
        for (size_t j = 0lu; j < shape.second.size(); j++)
            ss << "{" << ov::test::utils::vec2str(shape.second[j]) << "}";
        return ss;
    };

    axisUpdateShapeInShape shapes_desc;
    std::vector<InputShape> input_shapes;
    int64_t axis;
    ov::Shape indices_shape;
    std::vector<int64_t> indices_value;
    ov::element::Type model_type, indices_type;
    std::string target_device;
    std::tie(shapes_desc, indices_value, model_type, indices_type, target_device) = obj.param;
    std::tie(input_shapes, indices_shape, axis) = shapes_desc;

    std::ostringstream result;
    result << "InputShape=" << shapes_ss(input_shapes.at(0)).str() << "_";
    result << "IndicesShape=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "IndicesValue=" << ov::test::utils::vec2str(indices_value) << "_";
    result << "UpdateShape=" << shapes_ss(input_shapes.at(1)).str() << "_";
    result << "Axis=" << axis << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "idxType=" << indices_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ScatterUpdateLayerTest::SetUp() {
    axisUpdateShapeInShape shapes_desc;
    std::vector<InputShape> input_shapes;
    int64_t axis;
    ov::Shape indices_shape;
    std::vector<int64_t> indices_value;
    ov::element::Type model_type, indices_type;
    std::tie(shapes_desc, indices_value, model_type, indices_type, targetDevice) = this->GetParam();
    std::tie(input_shapes, indices_shape, axis) = shapes_desc;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(0));
    auto update_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(1));
    auto indices_const = std::make_shared<ov::op::v0::Constant>(indices_type, indices_shape, indices_value);
    auto axis_const =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{axis});
    auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(param, indices_const, update_param, axis_const);
    function = std::make_shared<ov::Model>(scatter->outputs(), ov::ParameterVector{param, update_param}, "ScatterUpdate");
}
}  // namespace test
}  // namespace ov
