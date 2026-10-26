// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "node_context.hpp"
#include "op_table.hpp"
#include "op_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace {
using namespace ov::frontend::gguf;

ov::Output<ov::Node> projection(std::shared_ptr<TensorMap>& tensors,
                                const std::shared_ptr<ConstantCache>& cache,
                                const ov::PartialShape& shape,
                                size_t width,
                                const std::map<std::string, ov::Any>& attributes = {}) {
    const std::string weight_name = "w" + std::to_string(width);
    std::vector<float> values(width * 8);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = (static_cast<int>(i % 7) - 3) / 8.f;
    (*tensors)[weight_name] = ov::op::v0::Constant::create(ov::element::f32, {width, 8}, values);
    const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_MUL_MAT",
        std::vector<ov_gguf_test::TensorDesc>{{weight_name, ov::element::f32, {1, 1, static_cast<int64_t>(width), 8}},
                                              {"x", tensors->at("x").get_element_type(), shape}},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
        attributes);
    return op::translate_mulmat(NodeContext(decoder, tensors, cache))[0];
}

std::shared_ptr<ov::Node> linear_node(const ov::Output<ov::Node>& output) {
    auto node = output.get_node_shared_ptr();
    if (node->get_type_name() == std::string("Unsqueeze"))
        node = node->get_input_node_shared_ptr(0);
    return node;
}
}  // namespace

TEST(GGUFSharedLinearInputs, PreservesBatchesDynamicShapesAndEmptyTokens) {
    const ov::PartialShape shape{2, 1, -1, -1};
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto tensors = std::make_shared<TensorMap>(TensorMap{{"x", x}});
    const auto y = projection(tensors, std::make_shared<ConstantCache>(), shape, 3);
    ASSERT_EQ(y.get_partial_shape(), (ov::PartialShape{2, 1, -1, 3}));
    ASSERT_EQ(y.get_node()->get_type_name(), std::string("MatMul"));
    const auto mm = linear_node(y);
    ASSERT_EQ(mm->input_value(0), x->output(0));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x});
    for (size_t tokens : {0, 1, 3}) {
        ov::Tensor input(ov::element::f32, {2, 1, tokens, 8});
        for (size_t i = 0; i < input.get_size(); ++i)
            input.data<float>()[i] = (static_cast<int>(i % 11) - 5) / 4.f;
        ov::Tensor output(ov::element::f32, {2, 1, tokens, 3});
        ov::TensorVector outputs{output};
        ASSERT_TRUE(model->evaluate(outputs, {input}));
        ASSERT_EQ(outputs[0].get_shape(), (ov::Shape{2, 1, tokens, 3}));
        for (size_t row = 0; row < 2 * tokens; ++row) {
            for (size_t n = 0; n < 3; ++n) {
                float expected = 0;
                for (size_t k = 0; k < 8; ++k)
                    expected += input.data<float>()[row * 8 + k] * (static_cast<int>((n * 8 + k) % 7) - 3) / 8.f;
                EXPECT_FLOAT_EQ(outputs[0].data<float>()[row * 3 + n], expected);
            }
        }
    }
}

TEST(GGUFSharedLinearInputs, SharesRotationAndTrailingConversionAcrossProjections) {
    const ov::PartialShape shape{2, 1, -1, 8};
    for (const auto type : {ov::element::f32, ov::element::f16}) {
        const auto x = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto tensors = std::make_shared<TensorMap>(TensorMap{{"x", x}});
        const auto cache = std::make_shared<ConstantCache>();
        const std::map<std::string, ov::Any> attributes{{"hadamard_block", int64_t{8}}};
        const auto q = projection(tensors, cache, shape, 3, attributes);
        const auto k = projection(tensors, cache, shape, 5, attributes);
        auto gate_attributes = attributes;
        gate_attributes["activation_rank"] = 3;
        const auto gate = projection(tensors, cache, shape, 7, gate_attributes);
        const auto shared_input = linear_node(q)->input_value(0);
        EXPECT_EQ(shared_input, linear_node(k)->input_value(0));
        EXPECT_NE(shared_input, linear_node(gate)->input_value(0));
        EXPECT_EQ(gate.get_partial_shape(), (ov::PartialShape{2, -1, 7}));
        EXPECT_EQ(cache->hadamard_outputs.size(), 1);
        EXPECT_EQ(cache->f32_activations.size(), type == ov::element::f16 ? 2 : 0);
        auto rotation = shared_input.get_node_shared_ptr();
        if (type == ov::element::f16)
            rotation = rotation->get_input_node_shared_ptr(0);
        EXPECT_EQ(rotation->get_type_name(), std::string("Reshape"));
    }
}

TEST(GGUFSharedLinearInputs, LeavesBroadcastWeightLayoutsUnchanged) {
    for (const auto& shape : {ov::PartialShape{2, 2, -1, 8}, ov::PartialShape{2, -1, -1, 8}}) {
        const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto tensors = std::make_shared<TensorMap>(TensorMap{{"x", x}});
        const auto cache = std::make_shared<ConstantCache>();
        const auto y = projection(tensors, cache, shape, 3, {{"hadamard_block", int64_t{8}}});
        EXPECT_EQ(linear_node(y)->get_input_partial_shape(0), shape);
        EXPECT_TRUE(cache->f32_activations.empty());
    }
    const ov::PartialShape shape{2, 1, -1, 8};
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto tensors = std::make_shared<TensorMap>(TensorMap{{"x", x}});
    const auto cache = std::make_shared<ConstantCache>();
    (*tensors)["w"] = ov::op::v0::Constant::create(ov::element::f32, {2, 3, 8}, {1.f});
    const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_MUL_MAT",
        std::vector<ov_gguf_test::TensorDesc>{{"w", ov::element::f32, {1, 2, 3, 8}}, {"x", ov::element::f32, shape}},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
        std::map<std::string, ov::Any>{{"hadamard_block", int64_t{8}}});
    const auto y = op::translate_mulmat(NodeContext(decoder, tensors, cache))[0];
    EXPECT_EQ(linear_node(y)->get_input_partial_shape(0), shape);
    EXPECT_EQ(y.get_partial_shape(), (ov::PartialShape{2, 2, -1, 3}));
    EXPECT_TRUE(cache->f32_activations.empty());
}

TEST(GGUFHadamardSharing, DistinguishesSignsDirectionLayoutAndConversionScope) {
    const ov::PartialShape shape{2, -1, 8};
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto tensors = std::make_shared<TensorMap>(TensorMap{{"x", x}});
    const auto cache = std::make_shared<ConstantCache>();
    const auto rotate = [&](const ov::Tensor& signs,
                            bool inverse,
                            const std::vector<int64_t>& heads,
                            const std::shared_ptr<ConstantCache>& conversion_cache) {
        const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
            "GGML_OP_MUL_MAT",
            std::vector<ov_gguf_test::TensorDesc>{{"x", ov::element::f32, shape}},
            std::vector<ov_gguf_test::TensorDesc>{},
            ov_gguf_test::TensorDesc{"y", ov::element::f32, shape},
            std::map<std::string, ov::Any>{{"hadamard_block", int64_t{8}},
                                           {"hadamard_signs", signs},
                                           {"hadamard_inverse", inverse},
                                           {"hadamard_gdn_heads", heads}});
        return apply_hadamard(NodeContext(decoder, tensors, conversion_cache), x);
    };
    ov::Tensor signs(ov::element::f32, {8});
    std::fill_n(signs.data<float>(), 8, 1.f);
    const auto first = rotate(signs, false, {}, cache);
    ov::Tensor same(ov::element::f32, {8});
    std::copy_n(signs.data<float>(), 8, same.data<float>());
    EXPECT_EQ(first, rotate(same, false, {}, cache));
    same.data<float>()[3] = -1.f;
    EXPECT_NE(first, rotate(same, false, {}, cache));
    EXPECT_NE(first, rotate(signs, true, {}, cache));
    EXPECT_NE(first, rotate(signs, false, {2, 2, 2}, cache));
    EXPECT_NE(first, rotate(signs, false, {}, std::make_shared<ConstantCache>()));
}
