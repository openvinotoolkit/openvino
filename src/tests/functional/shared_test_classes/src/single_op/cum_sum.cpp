// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/cum_sum.hpp"

namespace ov {
namespace test {
std::string CumSumLayerTest::getTestCaseName(const testing::TestParamInfo<cumSumParams>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    int64_t axis;
    bool exclusive, reverse;
    std::string targetDevice;
    std::tie(shapes, model_type, axis, exclusive, reverse, targetDevice) = obj.param;

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
    result << "Precision=" << model_type.get_type_name() << "_";
    result << "Axis=" << axis << "_";
    result << "Exclusive=" << (exclusive ? "TRUE" : "FALSE") << "_";
    result << "Reverse=" << (reverse ? "TRUE" : "FALSE") << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void CumSumLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    bool exclusive, reverse;
    int64_t axis;
    std::tie(shapes, model_type, axis, exclusive, reverse, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    const auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    const auto axis_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{axis});
    const auto cum_sum = std::make_shared<ov::op::v0::CumSum>(param, axis_node, exclusive, reverse);

    auto result = std::make_shared<ov::op::v0::Result>(cum_sum);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "cumsum");
}
}  // namespace test
}  // namespace ov
