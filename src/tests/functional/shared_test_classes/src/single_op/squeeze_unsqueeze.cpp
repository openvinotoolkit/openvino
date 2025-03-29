// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/squeeze_unsqueeze.hpp"

namespace ov {
namespace test {
std::string SqueezeUnsqueezeLayerTest::getTestCaseName(const testing::TestParamInfo<squeezeParams>& obj) {
    ov::element::Type model_type;
    ShapeAxesTuple shape_item;
    std::string targetDevice;
    ov::test::utils::SqueezeOpType op_type;
    std::tie(shape_item, op_type, model_type, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "IS=(";
    for (size_t i = 0lu; i < shape_item.first.size(); i++) {
        result << ov::test::utils::partialShape2str({shape_item.first[i].first})
               << (i < shape_item.first.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shape_item.first.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shape_item.first.size(); j++) {
            result << ov::test::utils::vec2str(shape_item.first[j].second[i]) << (j < shape_item.first.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "OpType=" << op_type << separator;
    result << "Axes=" << (shape_item.second.empty() ? "default" : ov::test::utils::vec2str(shape_item.second)) << separator;
    result << "modelType=" << model_type.to_string() << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void SqueezeUnsqueezeLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> input_shapes;
    std::vector<int> axes;
    ShapeAxesTuple shape_item;
    ov::test::utils::SqueezeOpType op_type;
    std::tie(shape_item, op_type, model_type, targetDevice) = GetParam();
    std::tie(input_shapes, axes) = shape_item;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    std::shared_ptr<ov::Node> op;

    if (axes.empty() && op_type == ov::test::utils::SqueezeOpType::SQUEEZE) {
        op = std::make_shared<ov::op::v0::Squeeze>(param);
    } else {
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
        if (op_type == op_type == ov::test::utils::SqueezeOpType::SQUEEZE)
            op = std::make_shared<ov::op::v0::Squeeze>(param, constant);
        else
            op = std::make_shared<ov::op::v0::Unsqueeze>(param, constant);
    }

    auto name = op_type == ov::test::utils::SqueezeOpType::SQUEEZE ? "Squeeze" : "Unsqueeze";

    function = std::make_shared<ov::Model>(op->outputs(), ov::ParameterVector{param}, name);
}
} // namespace test
} // namespace ov
