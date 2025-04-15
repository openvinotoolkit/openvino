// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/transpose.hpp"

namespace ov {
namespace test {
std::string TransposeLayerTest::getTestCaseName(const testing::TestParamInfo<transposeParams>& obj) {
    ov::element::Type modelType;
    std::vector<size_t> inputOrder;
    std::vector<InputShape> input_shapes;
    std::string targetDevice;
    std::tie(inputOrder, modelType, input_shapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "inputOrder=" << ov::test::utils::vec2str(inputOrder) << "_";
    result << "modelType=" << modelType.to_string() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void TransposeLayerTest::SetUp() {
    std::vector<size_t> input_order;
    std::vector<InputShape> input_shapes;
    ov::element::Type model_type;
    std::tie(input_order, model_type, input_shapes, targetDevice) = this->GetParam();

    init_input_shapes({input_shapes});

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    const auto in_order_shape = input_order.empty() ? ov::Shape({0}) : ov::Shape({inputDynamicShapes.front().size()});
    const auto input_order_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                         in_order_shape,
                                                                         input_order);
    const auto transpose = std::make_shared<ov::op::v1::Transpose>(param, input_order_const);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};
    function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "Transpose");
}
}  // namespace test
}  // namespace ov
