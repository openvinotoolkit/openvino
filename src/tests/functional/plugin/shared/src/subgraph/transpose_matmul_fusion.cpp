// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/transpose_matmul_fusion.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"

namespace ov {
namespace test {

std::string TransposeMatMulFusion::getTestCaseName(const testing::TestParamInfo<const char*> &obj) {
    return "device=" + std::string(obj.param);
}

void TransposeMatMulFusion::SetUp() {
    targetDevice = GetParam();

    ov::PartialShape shape1{1, 3, 128, 64};
    ov::PartialShape shape2{1, 3, 64, 128};

    InputShape input_shape1 = {shape1, {Shape{1, 3, 128, 64}}};
    InputShape input_shape2 = {shape2, {Shape{1, 3, 64, 128}}};
    init_input_shapes({input_shape1, input_shape2});

    const auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape1);
    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape2);
    const auto order = ov::op::v0::Constant::create(ov::element::i32, Shape{4}, {0, 1, 3, 2});
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(param1, order);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(param2, order);
    const auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose1, transpose2, false, false);
    const auto constant = op::v0::Constant::create(element::f32, Shape{1}, {9});
    const auto mul = std::make_shared<ov::op::v1::Multiply>(matmul, constant);
    function = std::make_shared<ov::Model>(mul, ov::ParameterVector{param1, param2});
}

void TransposeMatMulFusion::TearDown() {
    const auto model = compiledModel.get_runtime_model();

    int num_ops = 0;
    for (const auto& node : model->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type = rt_info.find("layerType")->second.as<std::string>();
        if (layer_type != "Reorder" && layer_type != "Const")
            num_ops++;
        EXPECT_NE(layer_type, "Transpose");
        EXPECT_NE(layer_type, "Permute");
    }
    ASSERT_EQ(num_ops, 5); // two Inputs, one Eltwise, one MatMul and one Output
}

}  // namespace test
}  // namespace ov
