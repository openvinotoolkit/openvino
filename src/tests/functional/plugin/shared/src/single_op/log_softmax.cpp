// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/single_op/log_softmax.hpp"
#include "openvino/op/log_softmax.hpp"

namespace ov {
namespace test {
std::string LogSoftmaxLayerTest::getTestCaseName(const testing::TestParamInfo<logSoftmaxLayerTestParams>& obj) {
    const auto& [model_type, shapes, axis, target_device] = obj.param;

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

    result << "modelType=" << model_type.get_type_name() << "_";
    result << "axis=" << axis << "_";
    result << "trgDev=" << target_device;

    return result.str();
}

void LogSoftmaxLayerTest::generate_inputs(const std::vector<ov::Shape>& target_shapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();
    auto& data_input = func_inputs[0];

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 5;
    in_data.range = 15;
    in_data.resolution = 1000;

    ov::Tensor data_tensor = ov::test::utils::create_and_fill_tensor(data_input.get_element_type(), data_input.get_shape(), in_data);
    inputs.insert({data_input.get_node_shared_ptr(), data_tensor});
}

void LogSoftmaxLayerTest::SetUp() {
    const auto& [model_type, shapes, axis, _targetDevice] = GetParam();
    targetDevice = _targetDevice;
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    const auto log_softmax = std::make_shared<ov::op::v5::LogSoftmax>(param, axis);

    auto result = std::make_shared<ov::op::v0::Result>(log_softmax);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "logSoftmax");
}
}  // namespace test
}  // namespace ov
