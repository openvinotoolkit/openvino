// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/clamp.hpp"

namespace ov {
namespace test {
std::string ClampLayerTest::getTestCaseName(const testing::TestParamInfo<clampParamsTuple>& obj) {
    std::vector<InputShape> shapes;
    std::pair<float, float> interval;
    ov::element::Type model_type;
    std::string target_device;

    std::tie(shapes, interval, model_type, target_device) = obj.param;

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
    result << "min=" << interval.first << "_";
    result << "max=" << interval.second << "_";
    result << "netPrc=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ClampLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    std::pair<float, float> interval;
    ov::element::Type model_type;
    std::tie(shapes, interval, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto input = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto clamp = std::make_shared<ov::op::v0::Clamp>(input, interval.first, interval.second);
    auto result = std::make_shared<ov::op::v0::Result>(clamp);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{input});
}
} // namespace test
} // namespace ov
