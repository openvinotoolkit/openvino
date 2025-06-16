// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/power.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/power.hpp"

namespace ov {
namespace test {
std::string PowerLayerTest::getTestCaseName(const testing::TestParamInfo<PowerParamsTuple> &obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string device_name;
    std::vector<float> power;
    std::tie(shapes, model_type, power, device_name) = obj.param;

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
    result << "Power=" << ov::test::utils::vec2str(power) << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << device_name << "_";
    return result.str();
}

void PowerLayerTest::SetUp() {
    abs_threshold = 0.04f;

    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::vector<float> power;
    std::tie(shapes, model_type, power, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto power_const = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{1}, power);
    auto pow = std::make_shared<ov::op::v1::Power>(param, power_const);

    function = std::make_shared<ov::Model>(pow, ov::ParameterVector{param}, "power");
}
} // namespace test
} // namespace ov
