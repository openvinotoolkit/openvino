// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "node_context.hpp"
#include "op_table.hpp"
#include "op_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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
    if (node->get_type_name() == std::string("Unsqueeze") || node->get_type_name() == std::string("Reshape"))
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
        EXPECT_EQ(cache->rank3_activations.size(), 1);
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

TEST(GGUFSharedLinearInputs, TernaryProjectionsExposeSupportedHadamardMatrices) {
    for (const int64_t block : {256, 512, 1024, 4096}) {
        SCOPED_TRACE(block);
        const ov::PartialShape shape{2, 1, -1, block};
        const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        const auto w = ov::op::v0::Constant::create(ov::element::u2, {16, static_cast<size_t>(block)}, {1});
        auto tensors = std::make_shared<TensorMap>(
            TensorMap{{"x", x}, {"w", std::make_shared<ov::op::v0::Convert>(w, ov::element::f32)}});
        const auto cache = std::make_shared<ConstantCache>();
        const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
            "GGML_OP_MUL_MAT",
            std::vector<ov_gguf_test::TensorDesc>{{"w", ov::element::f32, {1, 1, 16, block}},
                                                  {"x", ov::element::f32, shape}},
            std::vector<ov_gguf_test::TensorDesc>{},
            ov_gguf_test::TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
            std::map<std::string, ov::Any>{{"hadamard_block", block}});
        const NodeContext context(decoder, tensors, cache);
        const auto first = op::translate_mulmat(context)[0];
        const auto second = op::translate_mulmat(context)[0];
        EXPECT_EQ(first.get_partial_shape(), (ov::PartialShape{2, 1, -1, 16}));
        const auto fc = linear_node(first);
        EXPECT_EQ(fc->get_output_partial_shape(0), (ov::PartialShape{2, -1, 16}));
        EXPECT_EQ(fc->input_value(0), linear_node(second)->input_value(0));
        const auto rotation = fc->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0);
        ASSERT_EQ(rotation->get_type_name(), std::string("MatMul"));
        const auto matrix = ov::as_type_ptr<ov::op::v0::Constant>(rotation->get_input_node_shared_ptr(1));
        ASSERT_NE(matrix, nullptr);
        EXPECT_EQ(matrix->get_shape(), (ov::Shape{static_cast<size_t>(block), static_cast<size_t>(block)}));
        EXPECT_EQ(matrix, context.get_hadamard_matrix(ov::element::f32, block));
        EXPECT_EQ(cache->hadamard_matrices.size(), 1);
        EXPECT_EQ(cache->hadamard_outputs.size(), 1);
    }
}

TEST(GGUFSharedLinearInputs, TernaryProjectionRestoresSelectedTokenBatchLayout) {
    const ov::PartialShape shape{1, 1, -1, 8};
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto w = ov::op::v0::Constant::create(ov::element::u2, {3, 8}, {1});
    auto tensors = std::make_shared<TensorMap>(
        TensorMap{{"x", x}, {"w", std::make_shared<ov::op::v0::Convert>(w, ov::element::f32)}});
    const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_MUL_MAT",
        std::vector<ov_gguf_test::TensorDesc>{{"w", ov::element::f32, {1, 1, 3, 8}}, {"x", ov::element::f32, shape}},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
        std::map<std::string, ov::Any>{});
    const auto y = op::translate_mulmat(NodeContext(decoder, tensors, std::make_shared<ConstantCache>()))[0];
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x});
    x->set_partial_shape({1, -1, 1, 8});
    model->validate_nodes_and_infer_types();
    EXPECT_EQ(model->output().get_partial_shape(), (ov::PartialShape{1, -1, 1, 3}));
    ov::Tensor input(ov::element::f32, {1, 5, 1, 8});
    for (size_t i = 0; i < input.get_size(); ++i)
        input.data<float>()[i] = static_cast<float>(i);
    ov::TensorVector outputs{ov::Tensor(ov::element::f32, {1, 5, 1, 3})};
    ASSERT_TRUE(model->evaluate(outputs, {input}));
    EXPECT_EQ(outputs[0].get_shape(), (ov::Shape{1, 5, 1, 3}));
    for (size_t row = 0; row < 5; ++row)
        for (size_t column = 0; column < 3; ++column)
            EXPECT_FLOAT_EQ(outputs[0].data<float>()[row * 3 + column], 64.f * row + 28.f);
}

TEST(GGUFHadamardSharing, InverseRotationUsesSupportedMatrix) {
    using namespace ov::frontend::gguf;
    constexpr int64_t block = 256;
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, block});
    auto tensors = std::make_shared<TensorMap>(TensorMap{{"x", x}});
    const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_GET_ROWS",
        std::vector<ov_gguf_test::TensorDesc>{},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
        std::map<std::string, ov::Any>{{"hadamard_block", block}, {"hadamard_inverse", true}});
    const auto y = apply_hadamard(NodeContext(decoder, tensors, std::make_shared<ConstantCache>()), x);
    ASSERT_EQ(y.get_node()->get_type_name(), std::string("Reshape"));
    EXPECT_EQ(y.get_node()->get_input_node_shared_ptr(0)->get_type_name(), std::string("MatMul"));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x});
    ov::Tensor input(ov::element::f32, {2, 1, block});
    std::fill_n(input.data<float>(), input.get_size(), 1.f);
    ov::TensorVector output{ov::Tensor(ov::element::f32, {2, 1, block})};
    ASSERT_TRUE(model->evaluate(output, {input}));
    for (size_t i = 0; i < input.get_size(); ++i)
        EXPECT_FLOAT_EQ(output[0].data<float>()[i], i % block == 0 ? 16.f : 0.f);
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
