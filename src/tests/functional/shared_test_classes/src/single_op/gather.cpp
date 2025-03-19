// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/gather.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather.hpp"

namespace ov {
namespace test {
std::string GatherLayerTest::getTestCaseName(const testing::TestParamInfo<gatherParamsTuple> &obj) {
    int axis;
    std::vector<int> indices;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string device_name;
    std::tie(indices, indices_shape, axis, shapes, model_type, device_name) = obj.param;
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
    result << "indices=" << ov::test::utils::vec2str(indices) << "_";
    result << "indices_shape=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << device_name << "_";
    return result.str();
}

void GatherLayerTest::SetUp() {
    int axis;
    std::vector<int> indices;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(indices, indices_shape, axis, shapes, model_type, targetDevice) = GetParam();
    init_input_shapes(shapes);
    ASSERT_EQ(ov::shape_size(indices_shape), indices.size()) << "Indices vector size and provided indices shape doesn't fit each other";

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto indices_node = ov::op::v0::Constant::create(ov::element::i64, indices_shape, indices);
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape(), {axis});

    auto gather = std::make_shared<ov::op::v1::Gather>(param, indices_node, axis_node);
    auto result = std::make_shared<ov::op::v0::Result>(gather);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gather");
}

std::string Gather7LayerTest::getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj) {
    std::tuple<int, int> axis_batch_idx;
    std::vector<int> indices;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string device_name;
    std::tie(shapes, indices_shape, axis_batch_idx, model_type, device_name) = obj.param;
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
    result << "axis=" << std::get<0>(axis_batch_idx) << "_";
    result << "batch_idx=" << std::get<1>(axis_batch_idx) << "_";
    result << "indices_shape=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << device_name << "_";
    return result.str();
}

void Gather7LayerTest::SetUp() {
    std::tuple<int, int> axis_batch_idx;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(shapes, indices_shape, axis_batch_idx, model_type, targetDevice) = GetParam();
    init_input_shapes(shapes);

    int axis = std::get<0>(axis_batch_idx);
    int batch_idx = std::get<1>(axis_batch_idx);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    int axis_dim = targetStaticShapes[0][0][axis < 0 ? axis + targetStaticShapes[0][0].size() : axis];
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = axis_dim - 1;
    auto indices_node_tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, indices_shape, in_data);
    auto indices_node = std::make_shared<ov::op::v0::Constant>(indices_node_tensor);
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape(), {axis});

    auto gather = std::make_shared<ov::op::v7::Gather>(param, indices_node, axis_node, batch_idx);

    auto result = std::make_shared<ov::op::v0::Result>(gather);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gather");
}

std::string Gather8LayerTest::getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj) {
    return Gather7LayerTest::getTestCaseName(obj);
}

void Gather8LayerTest::SetUp() {
    std::tuple<int, int> axis_batch_idx;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(shapes, indices_shape, axis_batch_idx, model_type, targetDevice) = GetParam();
    init_input_shapes(shapes);

    int axis = std::get<0>(axis_batch_idx);
    int batch_idx = std::get<1>(axis_batch_idx);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    int axis_dim = targetStaticShapes[0][0][axis < 0 ? axis + targetStaticShapes[0][0].size() : axis];
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = -axis_dim;
    in_data.range = 2 * axis_dim;
    auto indices_node_tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, indices_shape, in_data);
    auto indices_node = std::make_shared<ov::op::v0::Constant>(indices_node_tensor);
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape(), {axis});

    auto gather = std::make_shared<ov::op::v8::Gather>(param, indices_node, axis_node, batch_idx);

    auto result = std::make_shared<ov::op::v0::Result>(gather);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gather");
}

std::string Gather8IndiceScalarLayerTest::getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj) {
    return Gather7LayerTest::getTestCaseName(obj);
}

void Gather8IndiceScalarLayerTest::SetUp() {
    std::tuple<int, int> axis_batch_idx;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(shapes, indices_shape, axis_batch_idx, model_type, targetDevice) = GetParam();
    init_input_shapes(shapes);

    int axis = std::get<0>(axis_batch_idx);
    int batch_idx = std::get<1>(axis_batch_idx);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto indices_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape(), {targetStaticShapes[0][0][axis] - 1});
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape(), {axis});

    auto gather = std::make_shared<ov::op::v8::Gather>(param, indices_node, axis_node, batch_idx);

    auto result = std::make_shared<ov::op::v0::Result>(gather);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gather");
}

std::string Gather8withIndicesDataLayerTest::getTestCaseName(const testing::TestParamInfo<gather8withIndicesDataParamsTuple>& obj) {
    gather7ParamsTuple basicParams;
    std::vector<int64_t> indicesData;
    std::tie(basicParams, indicesData) = obj.param;

    std::tuple<int, int> axis_batch_idx;
    std::vector<int> indices;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string device_name;
    std::tie(shapes, indices_shape, axis_batch_idx, model_type, device_name) = basicParams;

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
    result << "axis=" << std::get<0>(axis_batch_idx) << "_";
    result << "batch_idx=" << std::get<1>(axis_batch_idx) << "_";
    result << "indices_shape=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << device_name << "_";

    result << "indicesData=" << ov::test::utils::vec2str(indicesData) << "_";

    return result.str();
}

void Gather8withIndicesDataLayerTest::SetUp() {
    gather7ParamsTuple basicParams;
    std::vector<int64_t> indicesData;
    std::tie(basicParams, indicesData) = GetParam();

    std::tuple<int, int> axis_batch_idx;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(shapes, indices_shape, axis_batch_idx, model_type, targetDevice) = basicParams;
    init_input_shapes(shapes);

    int axis = std::get<0>(axis_batch_idx);
    int batch_idx = std::get<1>(axis_batch_idx);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    // create indices tensor and fill data
    ov::Tensor indices_node_tensor{ov::element::i64, indices_shape};
    auto indices_tensor_data = indices_node_tensor.data<int64_t>();
    for (size_t i = 0; i < shape_size(indices_shape); i++) {
        indices_tensor_data[i] = indicesData[i];
    }

    auto indices_node = std::make_shared<ov::op::v0::Constant>(indices_node_tensor);
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape(), {axis});

    auto gather = std::make_shared<ov::op::v8::Gather>(param, indices_node, axis_node, batch_idx);

    auto result = std::make_shared<ov::op::v0::Result>(gather);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gather");
}

// Gather String support
std::string GatherStringWithIndicesDataLayerTest::getTestCaseName(const testing::TestParamInfo<GatherStringParamsTuple>& obj) {
    const GatherStringParamsTuple& basicParams = obj.param;
    std::vector<int64_t> indicesData;
    std::vector<std::string> str_data;

    std::tuple<int, int> axis_batch_idx;
    std::vector<int> indices;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string device_name;
    std::tie(shapes, indices_shape, axis_batch_idx, model_type, device_name, indicesData, str_data) = basicParams;

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
    result << "axis=" << std::get<0>(axis_batch_idx) << "_";
    result << "batch_idx=" << std::get<1>(axis_batch_idx) << "_";
    result << "indices_shape=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << device_name << "_";
    result << "indicesData=" << ov::test::utils::vec2str(indicesData) << "_";

    return result.str();
}

void GatherStringWithIndicesDataLayerTest::SetUp() {
    const GatherStringParamsTuple& basicParams = GetParam();
    std::vector<int64_t> indicesData;
    std::tuple<int, int> axis_batch_idx;
    ov::Shape indices_shape;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(shapes, indices_shape, axis_batch_idx, model_type, targetDevice, indicesData, string_data) = basicParams;
    init_input_shapes(shapes);

    int axis = std::get<0>(axis_batch_idx);
    int batch_idx = std::get<1>(axis_batch_idx);
    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    // create indices tensor and fill data
    ov::Tensor indices_node_tensor{ov::element::i64, indices_shape};
    auto indices_tensor_data = indices_node_tensor.data<int64_t>();
    for (size_t i = 0; i < shape_size(indices_shape); ++i) {
        indices_tensor_data[i] = indicesData[i];
    }

    auto indices_node = std::make_shared<ov::op::v0::Constant>(indices_node_tensor);
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape(), {axis});
    auto gather = std::make_shared<ov::op::v8::Gather>(param, indices_node, axis_node, batch_idx);
    auto result = std::make_shared<ov::op::v0::Result>(gather);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gather");
}

void GatherStringWithIndicesDataLayerTest::generate_inputs(const std::vector<ov::Shape>& target_shapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();
    auto& data_input = func_inputs[0];
    inputs.insert({data_input.get_node_shared_ptr(), ov::Tensor(element::string, data_input.get_shape(), string_data.data())});
}

}  // namespace test
}  // namespace ov
