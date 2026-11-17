// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <tuple>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "gguf_writer.hpp"
#include "gtest/gtest.h"
#include "helper_ops/gguf_constant.hpp"
#include "node_context.hpp"
#include "op_table.hpp"
#include "op_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "pass/materialize_weights.hpp"
#include "quant/weights.hpp"

namespace {
using namespace ov::frontend::gguf;

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& model, const ov::TensorVector& inputs = {}) {
    ov::TensorVector outputs;
    for (const auto& output : model->outputs())
        outputs.emplace_back(output.get_element_type(), output.get_shape());
    EXPECT_TRUE(model->evaluate(outputs, inputs));
    return outputs;
}

void normalize(const std::shared_ptr<ov::Model>& model) {
    EXPECT_TRUE(pass::MaterializeGGUFWeights().run_on_model(model));
    EXPECT_EQ(count_ops_of_type<GGUFConstant>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::util::FrameworkNode>(model), 0);
    model->validate_nodes_and_infer_types();
}

struct TernarySource {
    GgufWeight source;
    std::vector<float> values;
};

TernarySource ternary(size_t rows, size_t seed, bool code_three = false) {
    constexpr size_t cols = 128;
    TernarySource result;
    result.source.data = ov::Tensor(ov::element::u8, {rows * 36});
    result.source.shape = {rows, cols};
    result.source.qtype = GGUF_TYPE_Q2_0;
    result.source.source_name = "blk.0.attn_" + std::to_string(seed) + ".weight";
    auto* bytes = result.source.data.data<uint8_t>();
    std::memset(bytes, 0, rows * 36);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t group = 0; group < 2; ++group) {
            const ov::float16 scale(static_cast<float>(row + group + seed + 1) * 0.25f);
            const auto bits = scale.to_bits();
            auto* block = bytes + (row * 2 + group) * 18;
            std::memcpy(block, &bits, 2);
            for (size_t col = 0; col < 64; ++col) {
                const uint8_t code = (row * 2 + col + group + seed) % (code_three ? 4 : 3);
                block[2 + col / 4] |= code << (2 * (col % 4));
                result.values.push_back((static_cast<int>(code) - 1) * static_cast<float>(scale));
            }
        }
    }
    return result;
}

void expect_values(const ov::Tensor& output, const std::vector<float>& values) {
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_size(), values.size());
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_FLOAT_EQ(output.data<const float>()[i], values[i]) << i;
}

std::shared_ptr<ov::Node> row_slice(const ov::Output<ov::Node>& input, int64_t start, int64_t stop, int64_t step = 1) {
    auto constant = [](int64_t value) {
        return ov::op::v0::Constant::create(ov::element::i64, {1}, {value});
    };
    return std::make_shared<ov::op::v8::Slice>(input, constant(start), constant(stop), constant(step), constant(0));
}

TEST(GGUFWeightNormalization, RawMetadataAndClonesRetainSourceStorage) {
    auto source = ternary(2, 0);
    const auto original = std::make_shared<GGUFConstant>(source.source);
    EXPECT_TRUE(ov::is_type<ov::op::util::FrameworkNode>(original));
    auto attrs = original->get_attrs();
    EXPECT_EQ(attrs.get_type_name(), "GGUFConstant");
    EXPECT_EQ(attrs.get_opset_name(), "gguf");
    attrs["diagnostic"] = "raw source";
    original->set_attrs(attrs);
    const auto clone = ov::as_type_ptr<GGUFConstant>(original->clone_with_new_inputs({}));
    ASSERT_TRUE(clone);
    EXPECT_TRUE(ov::is_type<ov::op::util::FrameworkNode>(clone));
    EXPECT_TRUE(clone->get_attrs() == original->get_attrs());
    EXPECT_EQ(clone->source().data.data(), source.source.data.data());
    EXPECT_EQ(clone->source().data.get_byte_size(), 72);
    EXPECT_EQ(clone->get_output_shape(0), (ov::Shape{2, 128}));
    EXPECT_EQ(clone->get_output_element_type(0), ov::element::f32);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{clone}, ov::ParameterVector{});
    source.source = {};
    normalize(model);
    expect_values(evaluate(model)[0], source.values);
    EXPECT_FALSE(pass::MaterializeGGUFWeights().run_on_model(model));
}

TEST(GGUFWeightNormalization, NestedConcatSlicesAndGatherFoldBeforeRepacking) {
    const auto a = ternary(2, 0), b = ternary(3, 1, true);
    const auto concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{std::make_shared<GGUFConstant>(a.source), std::make_shared<GGUFConstant>(b.source)},
        0);
    const auto slice = row_slice(concat, 1, 5);
    const auto indices = ov::op::v0::Constant::create(ov::element::i64, {3}, {3, 0, 3});
    const auto rows =
        std::make_shared<ov::op::v8::Gather>(slice, indices, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    std::vector<int64_t> order;
    for (int64_t i = 64; i < 128; ++i)
        order.push_back(i);
    for (int64_t i = 0; i < 64; ++i)
        order.push_back(i);
    const auto columns =
        std::make_shared<ov::op::v8::Gather>(rows,
                                             ov::op::v0::Constant::create(ov::element::i64, {128}, order),
                                             ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{columns}, ov::ParameterVector{});
    auto values = a.values;
    values.insert(values.end(), b.values.begin(), b.values.end());
    std::vector<float> expected;
    for (size_t row : {4, 1, 4})
        for (const auto column : order)
            expected.push_back(values[row * 128 + column]);
    normalize(model);
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Concat>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Gather>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Slice>(model), 0);
    expect_values(evaluate(model)[0], expected);
}

TEST(GGUFWeightNormalization, UnalignedColumnGatherPreservesLogicalValues) {
    const auto a = ternary(2, 0);
    std::vector<int64_t> order(128);
    std::iota(order.begin(), order.end(), 0);
    std::rotate(order.begin(), order.begin() + 1, order.end());
    const auto gather =
        std::make_shared<ov::op::v8::Gather>(std::make_shared<GGUFConstant>(a.source),
                                             ov::op::v0::Constant::create(ov::element::i64, {128}, order),
                                             ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{});
    normalize(model);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Gather>(model), 1);
    std::vector<float> expected;
    for (size_t row = 0; row < 2; ++row)
        for (auto col : order)
            expected.push_back(a.values[row * 128 + col]);
    expect_values(evaluate(model)[0], expected);
}

TEST(GGUFWeightNormalization, SharedProjectionsPreserveUnequalSizesTiesAndCodeThree) {
    const auto a = ternary(2, 0), b = ternary(3, 1, true), c = ternary(1, 2);
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 128});
    const auto shared = std::make_shared<GGUFConstant>(a.source);
    ov::OutputVector outputs;
    for (const auto& weight : ov::OutputVector{shared,
                                               std::make_shared<GGUFConstant>(b.source),
                                               std::make_shared<GGUFConstant>(c.source),
                                               shared})
        outputs.push_back(std::make_shared<ov::op::v0::MatMul>(input, weight, false, true));
    const auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});
    normalize(model);
    normalize_u2_zero_points(model);
    EXPECT_EQ(count_ops_of_type<ov::op::v0::MatMul>(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Subtract>(model), 1);
    ov::Tensor x(ov::element::f32, {2, 128});
    for (size_t i = 0; i < x.get_size(); ++i)
        x.data<float>()[i] = (static_cast<int>(i % 11) - 5) * 0.125f;
    const auto result = evaluate(model, {x});
    std::array<const TernarySource*, 4> sources{&a, &b, &c, &a};
    for (size_t out = 0; out < 4; ++out) {
        std::vector<float> expected;
        for (size_t token = 0; token < 2; ++token)
            for (size_t row = 0; row < sources[out]->source.shape[0]; ++row) {
                float sum = 0;
                for (size_t col = 0; col < 128; ++col)
                    sum += x.data<float>()[token * 128 + col] * sources[out]->values[row * 128 + col];
                expected.push_back(sum);
            }
        expect_values(result[out], expected);
    }
}

TEST(GGUFWeightNormalization, MixedWeightTypesKeepSeparateCompressedProjections) {
    const auto a = ternary(2, 0);
    ov::Tensor plain(ov::element::f32, {3, 128});
    std::fill_n(plain.data<float>(), plain.get_size(), 0.5f);
    auto b = GgufWeight(plain);
    b.source_name = "blk.0.attn_b.weight";
    const auto concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{std::make_shared<GGUFConstant>(a.source), std::make_shared<GGUFConstant>(b)},
        0);
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128});
    const auto mm = std::make_shared<ov::op::v0::MatMul>(input, concat, false, true);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{mm}, ov::ParameterVector{input});
    normalize(model);
    EXPECT_EQ(count_ops_of_type<ov::op::v0::MatMul>(model), 2);
    ov::Tensor x(ov::element::f32, {1, 128});
    std::fill_n(x.data<float>(), 128, 1.f);
    const auto result = evaluate(model, {x});
    std::vector<float> expected;
    for (size_t row = 0; row < 2; ++row)
        expected.push_back(std::accumulate(a.values.begin() + row * 128, a.values.begin() + (row + 1) * 128, 0.f));
    expected.insert(expected.end(), 3, 64.f);
    expect_values(result[0], expected);
}

TEST(GGUFWeightNormalization, PlainRowViewsAndConcatsPreserveValues) {
    for (const auto& shape : {ov::Shape{4}, ov::Shape{4, 3}}) {
        ov::Tensor data(ov::element::f32, shape);
        std::iota(data.data<float>(), data.data<float>() + data.get_size(), 1.f);
        auto raw = GgufWeight(data);
        raw.source_name = "blk.0.bias";
        const auto leaf = std::make_shared<GGUFConstant>(raw);
        const auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{leaf, row_slice(leaf, 1, 3)}, 0);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{});
        normalize(model);
        const auto row_size = data.get_size() / shape[0];
        std::vector<float> expected(data.data<float>(), data.data<float>() + data.get_size());
        expected.insert(expected.end(), data.data<float>() + row_size, data.data<float>() + 3 * row_size);
        EXPECT_EQ(count_ops_of_type<ov::op::v0::Concat>(model), 0);
        expect_values(evaluate(model)[0], expected);
    }
}

TEST(GGUFWeightNormalization, Q5KWeightAndBiasPoliciesRemainSeparate) {
    GgufWeight source;
    source.qtype = GGUF_TYPE_Q5_K;
    source.shape = {1, 256};
    source.data = ov::Tensor(ov::element::u8, {176});
    std::memset(source.data.data(), 1, 176);
    for (const bool bias_first : {false, true}) {
        source.source_name = "blk.0.ffn_up.weight";
        const auto weight = std::make_shared<GGUFConstant>(source);
        source.source_name = "blk.0.attn_q.bias";
        const auto bias = std::make_shared<GGUFConstant>(source);
        const auto inputs = bias_first ? ov::OutputVector{bias, weight} : ov::OutputVector{weight, bias};
        const auto model =
            std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Concat>(inputs, 0)},
                                        ov::ParameterVector{});
        const auto reference =
            std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Concat>(
                                            bias_first ? ov::OutputVector{bias->materialize(), weight->materialize()}
                                                       : ov::OutputVector{weight->materialize(), bias->materialize()},
                                            0)},
                                        ov::ParameterVector{});
        const auto expected = evaluate(reference)[0];
        normalize(model);
        EXPECT_EQ(count_ops_of_type<ov::op::v0::Concat>(model), 1);
        expect_values(
            evaluate(model)[0],
            std::vector<float>(expected.data<const float>(), expected.data<const float>() + expected.get_size()));
    }
}

TEST(GGUFWeightNormalization, LargePositiveSliceStepDoesNotOverflow) {
    const auto raw = ternary(3, 0);
    const auto slice = row_slice(std::make_shared<GGUFConstant>(raw.source), 1, 3, std::numeric_limits<int64_t>::max());
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{slice}, ov::ParameterVector{});
    normalize(model);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Slice>(model), 0);
    expect_values(evaluate(model)[0], std::vector<float>(raw.values.begin() + 128, raw.values.begin() + 256));
}

TEST(GGUFWeightNormalization, RankOneQuantizedViewsUseLogicalFallback) {
    auto raw = ternary(1, 0);
    raw.source.shape = {128};
    const auto model =
        std::make_shared<ov::Model>(ov::OutputVector{row_slice(std::make_shared<GGUFConstant>(raw.source), 1, 4)},
                                    ov::ParameterVector{});
    normalize(model);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Slice>(model), 1);
    expect_values(evaluate(model)[0], std::vector<float>(raw.values.begin() + 1, raw.values.begin() + 4));
}

TEST(GGUFWeightNormalization, DeferredTwoBitProjectionPreservesRankThreeLowering) {
    const ov::PartialShape shape{2, 1, -1, 256};
    for (const auto type : {GGUF_TYPE_Q2_0, GGUF_TYPE_PQ2_0, GGUF_TYPE_PTQ1_0, GGUF_TYPE_Q2_K}) {
        SCOPED_TRACE(type);
        auto source = ternary(3, 0).source;
        source.shape = {3, 256};
        source.qtype = type;
        const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto tensors =
            std::make_shared<TensorMap>(TensorMap{{"x", input}, {"w", std::make_shared<GGUFConstant>(source)}});
        const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
            "GGML_OP_MUL_MAT",
            std::vector<ov_gguf_test::TensorDesc>{{"w", ov::element::f32, {1, 1, 3, 256}},
                                                  {"x", ov::element::f32, shape}},
            std::vector<ov_gguf_test::TensorDesc>{},
            ov_gguf_test::TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
            std::map<std::string, ov::Any>{});
        const auto output = op::translate_mulmat(NodeContext(decoder, tensors, std::make_shared<ConstantCache>()))[0];
        EXPECT_EQ(output.get_partial_shape(), (ov::PartialShape{2, 1, -1, 3}));
        const auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(output.get_node()->get_input_node_shared_ptr(0));
        ASSERT_TRUE(mm);
        EXPECT_EQ(mm->get_output_partial_shape(0), (ov::PartialShape{2, -1, 3}));
    }
}

using QuantCase = std::tuple<GgufTensorType, size_t, size_t>;
class GGUFWeightFormatNormalization : public testing::TestWithParam<QuantCase> {};
TEST_P(GGUFWeightFormatNormalization, ConcatenatesRawRowsWithoutFloatWeightExpansion) {
    const auto type = std::get<0>(GetParam());
    const auto block = std::get<1>(GetParam());
    const auto bytes = std::get<2>(GetParam());
    auto source = [&](size_t rows, uint8_t seed) {
        GgufWeight weight;
        weight.qtype = type;
        weight.shape = {rows, 256};
        weight.source_name = "blk.0.attn_q.weight";
        weight.data = ov::Tensor(ov::element::u8, {rows * (256 / block) * bytes});
        std::memset(weight.data.data(), seed, weight.data.get_byte_size());
        return weight;
    };
    const auto a = source(2, 1);
    const auto b = source(3, 2);
    const auto concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{std::make_shared<GGUFConstant>(a), std::make_shared<GGUFConstant>(b)},
        0);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{});
    const auto reference = std::make_shared<ov::Model>(
        ov::OutputVector{
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{make_weight_node(a), make_weight_node(b)}, 0)},
        ov::ParameterVector{});
    const auto expected = evaluate(reference)[0];
    normalize(model);
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Concat>(model), 0);
    const auto actual = evaluate(model)[0];
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    for (size_t i = 0; i < expected.get_size(); ++i)
        EXPECT_FLOAT_EQ(actual.data<const float>()[i], expected.data<const float>()[i]) << i;
    bool combined = false;
    for (const auto& node : model->get_ordered_ops())
        if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            const auto& shape = constant->get_shape();
            combined |= shape.size() >= 2 && shape[0] == 5 && constant->get_element_type() != ov::element::f32;
        }
    EXPECT_TRUE(combined);
}
INSTANTIATE_TEST_SUITE_P(AllNativeQuantFormats,
                         GGUFWeightFormatNormalization,
                         testing::Values(QuantCase{GGUF_TYPE_Q1_0, 128, 18},
                                         QuantCase{GGUF_TYPE_Q4_0, 32, 18},
                                         QuantCase{GGUF_TYPE_Q4_1, 32, 20},
                                         QuantCase{GGUF_TYPE_Q5_0, 32, 22},
                                         QuantCase{GGUF_TYPE_Q5_1, 32, 24},
                                         QuantCase{GGUF_TYPE_Q8_0, 32, 34},
                                         QuantCase{GGUF_TYPE_Q8_1, 32, 36},
                                         QuantCase{GGUF_TYPE_Q2_K, 256, 84},
                                         QuantCase{GGUF_TYPE_Q3_K, 256, 110},
                                         QuantCase{GGUF_TYPE_Q4_K, 256, 144},
                                         QuantCase{GGUF_TYPE_Q5_K, 256, 176},
                                         QuantCase{GGUF_TYPE_Q6_K, 256, 210},
                                         QuantCase{GGUF_TYPE_Q8_K, 256, 292},
                                         QuantCase{GGUF_TYPE_MXFP4, 32, 17},
                                         QuantCase{GGUF_TYPE_Q2_0, 64, 18},
                                         QuantCase{GGUF_TYPE_PQ2_0, 128, 34},
                                         QuantCase{GGUF_TYPE_PTQ1_0, 128, 28}));

TEST(GGUFWeightNormalization, RawMappedSourceSurvivesLoaderRelease) {
    const auto generated = ternary(2, 0);
    std::vector<uint8_t> bytes(generated.source.data.get_byte_size());
    std::memcpy(bytes.data(), generated.source.data.data(), bytes.size());
    const auto path = ov::test::utils::generateTestFilePrefix() + ".gguf";
    ov_gguf_test::GgufWriter writer;
    writer.raw_tensor("blk.0.attn_q.weight", {128, 2}, GGUF_TYPE_Q2_0, bytes);
    ASSERT_TRUE(writer.write(path));
    std::shared_ptr<GGUFConstant> node;
    {
        const auto loaded = get_gguf_data(path);
        const auto& weights = std::get<1>(loaded);
        ASSERT_EQ(weights.size(), 1);
        const auto& source = weights.at("blk.0.attn_q.weight");
        EXPECT_EQ(source.shape, (ov::Shape{2, 128}));
        EXPECT_EQ(source.data.get_byte_size(), bytes.size());
        node = std::make_shared<GGUFConstant>(source);
    }
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    node.reset();
    normalize(model);
    expect_values(evaluate(model)[0], generated.values);
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    EXPECT_TRUE(std::filesystem::remove(path));
}

}  // namespace
