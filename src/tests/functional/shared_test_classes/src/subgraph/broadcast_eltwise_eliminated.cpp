// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/broadcast_eltwise_eliminated.hpp"

namespace ov {
namespace test {

std::string BroadcastEltwiseEliminated::getTestCaseName(const testing::TestParamInfo<const char*> &obj) {
    return "device=" + std::string(obj.param);
}

void BroadcastEltwiseEliminated::SetUp() {
    targetDevice = GetParam();

    ov::PartialShape shape{-1, 3, 10, 10};

    InputShape input_shape = {shape, {Shape{1, 3, 10, 10}}};
    init_input_shapes({input_shape});

    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(param);
    const auto constant = op::v0::Constant::create(element::f32, Shape{1}, {9});
    const auto bcast = std::make_shared<ov::op::v3::Broadcast>(constant, shapeof);
    const auto mul = std::make_shared<ov::op::v1::Multiply>(param, bcast);
    function = std::make_shared<ov::Model>(mul, ov::ParameterVector{param});
}

void BroadcastEltwiseEliminated::TearDown() {
    const auto model = compiledModel.get_runtime_model();

    int num_ops = 0;
    for (const auto& node : model->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type = rt_info.find("layerType")->second.as<std::string>();
        if (layer_type != "Reorder" && layer_type != "Const")
            num_ops++;
        EXPECT_NE(layer_type, "Broadcast");
    }
    ASSERT_EQ(num_ops, 3); // one Input, one Eltwise and one Output
}

}  // namespace test
}  // namespace ov
