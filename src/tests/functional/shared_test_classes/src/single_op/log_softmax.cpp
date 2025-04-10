// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/log_softmax.hpp"

namespace ov {
namespace test {
std::string LogSoftmaxLayerTest::getTestCaseName(const testing::TestParamInfo<logSoftmaxLayerTestParams>& obj) {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    int64_t axis;
    std::string target_device;
    std::tie(model_type, shapes, axis, target_device) = obj.param;

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

void LogSoftmaxLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    int64_t axis;

    std::tie(model_type, shapes, axis, targetDevice) = GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    const auto log_softmax = std::make_shared<ov::op::v5::LogSoftmax>(param, axis);

    auto result = std::make_shared<ov::op::v0::Result>(log_softmax);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "logSoftmax");
}
}  // namespace test
}  // namespace ov
