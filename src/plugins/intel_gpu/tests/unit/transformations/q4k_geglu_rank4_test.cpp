// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/q4k_geglu_rank4.hpp"

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/runtime/tensor.hpp"

TEST(Q4KGeGLURank4, ExpandsOnlyMarkedProjections) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 1, -1, 8});
    const auto rank3 = std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, -1, 8}), true);
    const auto converted = std::make_shared<ov::op::v0::Convert>(rank3, ov::element::f32);
    const auto weights = ov::op::v0::Constant::create(ov::element::f32, {4, 8}, {0.25f});
    const auto marked = std::make_shared<ov::op::v0::MatMul>(converted, weights, false, true);
    marked->get_rt_info()["gguf_gpu_rank4_hint"] = true;
    const auto marked_up = std::make_shared<ov::op::v0::MatMul>(converted, weights, false, true);
    marked_up->get_rt_info()["gguf_gpu_rank4_hint"] = true;
    const auto unmarked = std::make_shared<ov::op::v0::MatMul>(rank3, weights, false, true);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{marked, marked_up, unmarked}, ov::ParameterVector{input});

    ov::Tensor activation(ov::element::f32, {2, 1, 3, 8});
    for (size_t i = 0; i < activation.get_size(); ++i)
        activation.data<float>()[i] = static_cast<float>(static_cast<int>(i) - 10) / 8.f;
    ov::TensorVector expected{ov::Tensor(ov::element::f32, {2, 3, 4}), ov::Tensor(ov::element::f32, {2, 3, 4}), ov::Tensor(ov::element::f32, {2, 3, 4})};
    ASSERT_TRUE(model->evaluate(expected, {activation}));

    ov::intel_gpu::Q4KGeGLURank4 pass;
    EXPECT_TRUE(pass.run_on_model(model));
    EXPECT_EQ(marked->get_input_partial_shape(0), (ov::PartialShape{2, 1, -1, 8}));
    EXPECT_EQ(marked->get_output_partial_shape(0), (ov::PartialShape{2, 1, -1, 4}));
    EXPECT_EQ(marked_up->get_input_partial_shape(0), (ov::PartialShape{2, 1, -1, 8}));
    EXPECT_EQ(marked->input_value(0).get_node_shared_ptr(), marked_up->input_value(0).get_node_shared_ptr());
    EXPECT_EQ(unmarked->get_input_partial_shape(0), (ov::PartialShape{2, -1, 8}));
    EXPECT_EQ(unmarked->get_output_partial_shape(0), (ov::PartialShape{2, -1, 4}));

    ov::TensorVector actual{ov::Tensor(ov::element::f32, {2, 1, 3, 4}), ov::Tensor(ov::element::f32, {2, 1, 3, 4}), ov::Tensor(ov::element::f32, {2, 3, 4})};
    ASSERT_TRUE(model->evaluate(actual, {activation}));
    for (size_t i = 0; i < expected[0].get_size(); ++i)
        EXPECT_FLOAT_EQ(actual[0].data<const float>()[i], expected[0].data<const float>()[i]);
    for (size_t i = 0; i < expected[1].get_size(); ++i)
        EXPECT_FLOAT_EQ(actual[1].data<const float>()[i], expected[1].data<const float>()[i]);
    for (size_t i = 0; i < expected[2].get_size(); ++i)
        EXPECT_FLOAT_EQ(actual[2].data<const float>()[i], expected[2].data<const float>()[i]);
}

TEST(Q4KGeGLURank4, ExpandsFlattenedInputWhenOriginalReshapeIsUnavailable) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 8});
    const auto weights = ov::op::v0::Constant::create(ov::element::f32, {4, 8}, {0.25f});
    const auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, true);
    matmul->get_rt_info()["gguf_gpu_rank4_hint"] = true;
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});

    ov::Tensor activation(ov::element::f32, {2, 3, 8});
    for (size_t i = 0; i < activation.get_size(); ++i)
        activation.data<float>()[i] = static_cast<float>(static_cast<int>(i) - 10) / 8.f;
    ov::TensorVector expected{ov::Tensor(ov::element::f32, {2, 3, 4})};
    ASSERT_TRUE(model->evaluate(expected, {activation}));

    ov::intel_gpu::Q4KGeGLURank4 pass;
    EXPECT_TRUE(pass.run_on_model(model));
    EXPECT_EQ(matmul->get_output_partial_shape(0), (ov::PartialShape{2, 1, -1, 4}));

    ov::TensorVector actual{ov::Tensor(ov::element::f32, {2, 1, 3, 4})};
    ASSERT_TRUE(model->evaluate(actual, {activation}));
    for (size_t i = 0; i < expected[0].get_size(); ++i)
        EXPECT_FLOAT_EQ(actual[0].data<const float>()[i], expected[0].data<const float>()[i]);
}
