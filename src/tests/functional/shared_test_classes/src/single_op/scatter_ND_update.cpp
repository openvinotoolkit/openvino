// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/scatter_ND_update.hpp"

namespace ov {
namespace test {
std::string ScatterNDUpdateLayerTest::getTestCaseName(const testing::TestParamInfo<scatterNDUpdateParamsTuple> &obj) {
    auto shapes_ss = [](const InputShape& shape) {
        std::stringstream ss;
        ss << "_IS=(" << ov::test::utils::partialShape2str({shape.first}) << ")_TS=";
        for (size_t j = 0lu; j < shape.second.size(); j++)
            ss << "{" << ov::test::utils::vec2str(shape.second[j]) << "}";
        return ss;
    };

    scatterNDUpdateSpecParams shapes_desc;
    std::vector<InputShape> input_shapes;
    ov::Shape indices_shape;
    std::vector<int> indices_value;
    ov::element::Type model_type, indices_type;
    std::string target_device;
    std::tie(shapes_desc, model_type, indices_type, target_device) = obj.param;
    std::tie(input_shapes, indices_shape, indices_value) = shapes_desc;

    std::ostringstream result;
    result << "InputShape=" << shapes_ss(input_shapes.at(0)).str() << "_";
    result << "IndicesShape=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "IndicesValue=" << ov::test::utils::vec2str(indices_value) << "_";
    result << "UpdateShape=" << shapes_ss(input_shapes.at(1)).str() << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "idxType=" << indices_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ScatterNDUpdateLayerTest::SetUp() {
    scatterNDUpdateSpecParams shapes_desc;
    std::vector<InputShape> input_shapes;
    ov::Shape indices_shape;
    std::vector<int> indices_value;
    ov::element::Type model_type, indices_type;
    std::tie(shapes_desc, model_type, indices_type, targetDevice) = this->GetParam();
    std::tie(input_shapes, indices_shape, indices_value) = shapes_desc;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(0));
    auto update_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(1));
    auto indices_const = std::make_shared<ov::op::v0::Constant>(indices_type, indices_shape, indices_value);
    auto scatter_nd = std::make_shared<ov::op::v3::ScatterNDUpdate>(param, indices_const, update_param);
    function = std::make_shared<ov::Model>(scatter_nd->outputs(), ov::ParameterVector{param, update_param}, "ScatterNDUpdate");
}

std::string ScatterNDUpdate15LayerTest::getTestCaseName(const testing::TestParamInfo<scatterNDUpdate15ParamsTuple> &obj) {
    auto shapes_ss = [](const InputShape& shape) {
        std::stringstream ss;
        ss << "_IS=(" << ov::test::utils::partialShape2str({shape.first}) << ")_TS=";
        for (size_t j = 0lu; j < shape.second.size(); j++)
            ss << "{" << ov::test::utils::vec2str(shape.second[j]) << "}";
        return ss;
    };

    scatterNDUpdateSpecParams shapes_desc;
    std::vector<InputShape> input_shapes;
    ov::Shape indices_shape;
    std::vector<int> indices_value;
    ov::element::Type model_type, indices_type;
    ov::op::v15::ScatterNDUpdate::Reduction reduceMode;
    std::string target_device;
    std::tie(shapes_desc, reduceMode, model_type, indices_type, target_device) = obj.param;
    std::tie(input_shapes, indices_shape, indices_value) = shapes_desc;

    std::ostringstream result;
    result << "InputShape=" << shapes_ss(input_shapes.at(0)).str() << "_";
    result << "IndicesShape=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "IndicesValue=" << ov::test::utils::vec2str(indices_value) << "_";
    result << "UpdateShape=" << shapes_ss(input_shapes.at(1)).str() << "_";
    result << "ReduceMode=" << as_string(reduceMode) << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "idxType=" << indices_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ScatterNDUpdate15LayerTest::SetUp() {
    scatterNDUpdateSpecParams shapes_desc;
    std::vector<InputShape> input_shapes;
    ov::Shape indices_shape;
    std::vector<int> indices_value;
    ov::element::Type model_type, indices_type;
    ov::op::v15::ScatterNDUpdate::Reduction reduceMode;
    std::tie(shapes_desc, reduceMode, model_type, indices_type, targetDevice) = this->GetParam();
    std::tie(input_shapes, indices_shape, indices_value) = shapes_desc;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(0));
    auto update_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(1));
    auto indices_const = std::make_shared<ov::op::v0::Constant>(indices_type, indices_shape, indices_value);
    auto scatter_nd = std::make_shared<ov::op::v15::ScatterNDUpdate>(param, indices_const, update_param, reduceMode);
    function = std::make_shared<ov::Model>(scatter_nd->outputs(), ov::ParameterVector{param, update_param}, "ScatterNDUpdate");
}
}  // namespace test
}  // namespace ov

