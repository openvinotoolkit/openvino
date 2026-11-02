// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "quant/deferred_weights.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
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
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "pass/materialize_weights.hpp"
#include "quant/weights.hpp"

namespace {

using namespace ov::frontend::gguf;

struct U2Weight {
    WeightTensors tensors;
    std::vector<float> values;
    std::shared_ptr<std::atomic<size_t>> repacks = std::make_shared<std::atomic<size_t>>(0);
    std::shared_ptr<std::array<void*, 3>> destinations = std::make_shared<std::array<void*, 3>>();
};

U2Weight make_u2_weight(size_t rows, size_t seed, bool code_three = false, size_t group_size = 64) {
    constexpr size_t cols = 128;
    const size_t groups = cols / group_size;
    U2Weight result;
    std::vector<uint8_t> codes(rows * cols / 4, 0);
    std::vector<ov::float16> scales(rows * groups);
    result.values.resize(rows * cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t group = 0; group < groups; ++group) {
            scales[row * groups + group] = static_cast<float>(row + group + seed + 1) * 0.25f;
        }
        for (size_t col = 0; col < cols; ++col) {
            const size_t index = row * cols + col;
            const uint8_t code = (row * 2 + col + seed) % (code_three ? 4 : 3);
            codes[index / 4] |= code << (2 * (index % 4));
            result.values[index] =
                (static_cast<int>(code) - 1) * static_cast<float>(scales[row * groups + col / group_size]);
        }
    }
    const std::array<DeferredWeightTensor, 3> components{
        {{ov::element::u2, {rows, cols}}, {ov::element::f16, {rows, groups}}, {ov::element::u8, {rows, groups}}}};
    result.tensors = defer_weight_tensors(
        components,
        [codes, scales, count = result.repacks, targets = result.destinations](WeightTensors& destination) {
            ++*count;
            *targets = {destination.weight.data(), destination.scales.data(), destination.zero_point.data()};
            std::memcpy(destination.weight.data(), codes.data(), codes.size());
            std::memcpy(destination.scales.data(), scales.data(), scales.size() * sizeof(ov::float16));
            std::memset(destination.zero_point.data(), 1, scales.size());
        });
    return result;
}

std::shared_ptr<GGUFConstant> placeholder(const U2Weight& weight,
                                          const std::string& name,
                                          GgufTensorType type = GGUF_TYPE_Q2_0) {
    return std::make_shared<GGUFConstant>(weight.tensors, type, name);
}

template <typename T>
size_t count_nodes(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        count += ov::is_type<T>(node);
    }
    return count;
}

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& model, const ov::TensorVector& inputs = {}) {
    ov::TensorVector outputs;
    for (const auto& output : model->outputs()) {
        outputs.emplace_back(output.get_element_type(), output.get_shape());
    }
    EXPECT_TRUE(model->evaluate(outputs, inputs));
    return outputs;
}

void expect_values(const ov::Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.get_element_type(), ov::element::f32);
    ASSERT_EQ(tensor.get_size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor.data<const float>()[i], values[i]) << "element " << i;
    }
}

ov::Tensor input_tensor(const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        tensor.data<float>()[i] = (static_cast<int>(i % 11) - 5) * 0.125f;
    }
    return tensor;
}

std::vector<float> expected_projection(const ov::Tensor& input, const U2Weight& weight) {
    const size_t cols = input.get_shape().back();
    const size_t rows = weight.values.size() / cols;
    std::vector<float> values(input.get_size() / cols * rows);
    for (size_t token = 0; token < input.get_size() / cols; ++token) {
        for (size_t row = 0; row < rows; ++row) {
            float sum = 0;
            for (size_t col = 0; col < cols; ++col) {
                sum += input.data<const float>()[token * cols + col] * weight.values[row * cols + col];
            }
            values[token * rows + row] = sum;
        }
    }
    return values;
}

void materialize(const std::shared_ptr<ov::Model>& model) {
    EXPECT_TRUE(pass::MaterializeGGUFWeights().run_on_model(model));
    EXPECT_EQ(count_nodes<GGUFConstant>(model), 0);
    model->validate_nodes_and_infer_types();
}

TEST(GGUFDeferredWeights, MetadataAndCopiedHandlesDoNotMaterialize) {
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto tensor = defer_tensor(ov::element::f32, {2, 3}, [calls] {
        ++*calls;
        ov::Tensor value(ov::element::f32, {2, 3});
        std::fill_n(value.data<float>(), 6, 7.f);
        return value;
    });
    auto alias = tensor;
    EXPECT_EQ(alias.get_element_type(), ov::element::f32);
    EXPECT_EQ(alias.get_shape(), (ov::Shape{2, 3}));
    EXPECT_EQ(alias.get_size(), 6);
    EXPECT_EQ(alias.get_byte_size(), 24);
    EXPECT_EQ(alias.get_strides(), (ov::Strides{12, 4}));
    EXPECT_TRUE(alias.is_continuous());
    EXPECT_FALSE(is_materialized(tensor));
    EXPECT_EQ(*calls, 0);
    EXPECT_EQ(alias.data<const float>()[5], 7.f);
    EXPECT_EQ(tensor.data(), alias.data());
    EXPECT_EQ(*calls, 1);
    EXPECT_EQ(get_materialization_count(tensor), 1);
}

TEST(GGUFDeferredWeights, ConcurrentAliasesMaterializeOnce) {
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    const auto tensor = defer_tensor(ov::element::f32, {4}, [calls] {
        ++*calls;
        ov::Tensor value(ov::element::f32, {4});
        std::fill_n(value.data<float>(), 4, 5.f);
        return value;
    });
    std::promise<void> start;
    const auto ready = start.get_future().share();
    std::array<const float*, 8> pointers{};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < pointers.size(); ++i) {
        threads.emplace_back([alias = tensor, ready, &pointers, i] {
            ready.wait();
            pointers[i] = alias.data<const float>();
        });
    }
    start.set_value();
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto* pointer : pointers) {
        ASSERT_EQ(pointer, pointers[0]);
        EXPECT_EQ(pointer[3], 5.f);
    }
    EXPECT_EQ(*calls, 1);
    EXPECT_EQ(get_materialization_count(tensor), 1);
}

TEST(GGUFDeferredWeights, ComponentsShareOneConcurrentRepack) {
    auto weight = make_u2_weight(2, 0);
    EXPECT_EQ(weight.tensors.weight.get_shape(), (ov::Shape{2, 128}));
    EXPECT_EQ(weight.tensors.scales.get_shape(), (ov::Shape{2, 2}));
    EXPECT_EQ(weight.tensors.weight.get_byte_size(), 64);
    EXPECT_FALSE(is_materialized(weight.tensors.scales));
    std::promise<void> start;
    const auto ready = start.get_future().share();
    const std::array<ov::Tensor, 3> components{weight.tensors.weight, weight.tensors.scales, weight.tensors.zero_point};
    std::array<const void*, 6> pointers{};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < pointers.size(); ++i) {
        threads.emplace_back([tensor = components[i % 3], ready, &pointers, i] {
            ready.wait();
            pointers[i] = tensor.data();
        });
    }
    start.set_value();
    for (auto& thread : threads) {
        thread.join();
    }
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(pointers[i], pointers[i + 3]);
        EXPECT_TRUE(is_materialized(components[i]));
        EXPECT_EQ(get_materialization_count(components[i]), 1);
    }
    EXPECT_EQ(*weight.repacks, 1);
}

TEST(GGUFDeferredWeights, NestedRowConcatRepacksIntoFinalBuffers) {
    auto first = make_u2_weight(2, 0);
    auto second = make_u2_weight(3, 1);
    auto third = make_u2_weight(1, 2);
    const auto pair = concat_deferred_rows(first.tensors, second.tensors);
    ASSERT_TRUE(pair);
    const auto combined = concat_deferred_rows(*pair, third.tensors);
    ASSERT_TRUE(combined);
    EXPECT_EQ(combined->weight.get_shape(), (ov::Shape{6, 128}));
    EXPECT_EQ(*first.repacks + *second.repacks + *third.repacks, 0);
    const std::array<ov::Tensor, 3> outputs{combined->weight, combined->scales, combined->zero_point};
    const std::array<size_t, 3> bytes_per_row{32, 4, 2};
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto* pointer = static_cast<const uint8_t*>(outputs[i].data());
        EXPECT_EQ((*first.destinations)[i], pointer);
        EXPECT_EQ((*second.destinations)[i], pointer + 2 * bytes_per_row[i]);
        EXPECT_EQ((*third.destinations)[i], pointer + 5 * bytes_per_row[i]);
        EXPECT_EQ(get_materialization_count(outputs[i]), 1);
    }
    for (const auto* source : {&first, &second, &third}) {
        EXPECT_EQ(*source->repacks, 1);
        EXPECT_FALSE(is_materialized(source->tensors.weight));
        EXPECT_FALSE(is_materialized(source->tensors.scales));
        EXPECT_FALSE(is_materialized(source->tensors.zero_point));
    }
    EXPECT_FALSE(is_materialized(pair->weight));
    auto expected = first.values;
    expected.insert(expected.end(), second.values.begin(), second.values.end());
    expected.insert(expected.end(), third.values.begin(), third.values.end());
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{make_weight_node(*combined, GGUF_TYPE_Q2_0)},
                                                   ov::ParameterVector{});
    expect_values(evaluate(model)[0], expected);
}

TEST(GGUFDeferredWeights, IncompatibleLayoutsAndUnalignedRowsStayLazy) {
    const auto first = make_u2_weight(2, 0);
    const std::array<DeferredWeightTensor, 3> compatible{
        {{ov::element::u2, {1, 128}}, {ov::element::f16, {1, 2}}, {ov::element::u8, {1, 2}}}};
    auto calls = std::make_shared<size_t>(0);
    for (size_t scenario = 0; scenario < 5; ++scenario) {
        SCOPED_TRACE(scenario);
        auto components = compatible;
        switch (scenario) {
        case 0:
            components[0].shape.back() = 64;
            break;
        case 1:
            components[1].type = ov::element::f32;
            break;
        case 2:
            components[2] = {};
            break;
        case 3:
            components[1].shape.back() = 1;
            break;
        default:
            components[1].shape[0] = 2;
        }
        const auto other = defer_weight_tensors(components, [calls](WeightTensors&) {
            ++*calls;
        });
        EXPECT_FALSE(concat_deferred_rows(first.tensors, other));
    }
    const std::array<DeferredWeightTensor, 3> unaligned{{{ov::element::u2, {1, 3}}, {}, {}}};
    const auto short_row = defer_weight_tensors(unaligned, [calls](WeightTensors&) {
        ++*calls;
    });
    EXPECT_FALSE(concat_deferred_rows(short_row, short_row));
    EXPECT_EQ(*calls, 0);
    EXPECT_EQ(*first.repacks, 0);
}

TEST(GGUFMaterializeWeights, CloneRetainsLazySharedStorage) {
    const auto weight = make_u2_weight(2, 0);
    const auto node = placeholder(weight, "blk.0.attn_q.weight");
    const auto clone = ov::as_type_ptr<GGUFConstant>(node->clone_with_new_inputs({}));
    ASSERT_TRUE(clone);
    EXPECT_EQ(clone->get_output_shape(0), (ov::Shape{2, 128}));
    EXPECT_EQ(clone->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(*weight.repacks, 0);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node, clone}, ov::ParameterVector{});
    materialize(model);
    const auto outputs = evaluate(model);
    expect_values(outputs[0], weight.values);
    expect_values(outputs[1], weight.values);
    EXPECT_EQ(*weight.repacks, 1);
}

TEST(GGUFMaterializeWeights, CallbackPlaceholderCloneDoesNotAllocate) {
    auto calls = std::make_shared<size_t>(0);
    const auto node = std::make_shared<GGUFConstant>(ov::Shape{2}, ov::element::f32, GGUF_TYPE_F32, "bias", [calls] {
        ++*calls;
        return ov::op::v0::Constant::create(ov::element::f32, {2}, {3.f, -2.f});
    });
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    const auto clone = model->clone();
    EXPECT_EQ(*calls, 0);
    EXPECT_EQ(count_nodes<GGUFConstant>(clone), 1);
    materialize(clone);
    expect_values(evaluate(clone)[0], {3.f, -2.f});
    EXPECT_EQ(*calls, 1);
    EXPECT_EQ(count_nodes<GGUFConstant>(model), 1);
}

TEST(GGUFMaterializeWeights, DeferredTwoBitProjectionPreservesRankThreeLowering) {
    const ov::PartialShape shape{2, 1, -1, 128};
    for (const auto qtype : {GGUF_TYPE_Q2_0, GGUF_TYPE_PQ2_0, GGUF_TYPE_PTQ1_0, GGUF_TYPE_Q2_K}) {
        SCOPED_TRACE(qtype);
        auto calls = std::make_shared<size_t>(0);
        const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        const auto weight = std::make_shared<GGUFConstant>(ov::Shape{3, 128}, ov::element::f32, qtype, "w", [calls] {
            ++*calls;
            return ov::op::v0::Constant::create(ov::element::f32, {3, 128}, {1.f});
        });
        auto tensors = std::make_shared<TensorMap>(TensorMap{{"x", input}, {"w", weight}});
        const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
            "GGML_OP_MUL_MAT",
            std::vector<ov_gguf_test::TensorDesc>{{"w", ov::element::f32, {1, 1, 3, 128}},
                                                  {"x", ov::element::f32, shape}},
            std::vector<ov_gguf_test::TensorDesc>{},
            ov_gguf_test::TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
            std::map<std::string, ov::Any>{});
        const NodeContext context(decoder, tensors, std::make_shared<ConstantCache>());
        const auto output = op::translate_mulmat(context)[0];
        EXPECT_EQ(output.get_partial_shape(), (ov::PartialShape{2, 1, -1, 3}));
        const auto projection = ov::as_type_ptr<ov::op::v0::MatMul>(output.get_node()->get_input_node_shared_ptr(0));
        ASSERT_TRUE(projection);
        EXPECT_EQ(projection->get_output_partial_shape(0), (ov::PartialShape{2, -1, 3}));
        EXPECT_EQ(*calls, 0);
    }
}

TEST(GGUFMaterializeWeights, NestedRowConcatRemovesEveryPlaceholder) {
    const auto first = make_u2_weight(2, 0);
    const auto second = make_u2_weight(3, 1);
    const auto third = make_u2_weight(1, 2);
    const auto pair = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{placeholder(first, "blk.0.attn_q.weight"), placeholder(second, "blk.0.attn_k.weight")},
        0);
    const auto concat =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{pair, placeholder(third, "blk.0.attn_v.weight")}, 0);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{});
    materialize(model);
    EXPECT_EQ(count_nodes<ov::op::v0::Concat>(model), 0);
    auto expected = first.values;
    expected.insert(expected.end(), second.values.begin(), second.values.end());
    expected.insert(expected.end(), third.values.begin(), third.values.end());
    expect_values(evaluate(model)[0], expected);
    for (const auto* source : {&first, &second, &third}) {
        EXPECT_EQ(*source->repacks, 1);
        EXPECT_FALSE(is_materialized(source->tensors.weight));
    }
    EXPECT_FALSE(pass::MaterializeGGUFWeights().run_on_model(model));
}

TEST(GGUFMaterializeWeights, SharedInputFusionPreservesUnequalProjectionSizesAndTiedUses) {
    const auto first = make_u2_weight(2, 0);
    const auto second = make_u2_weight(3, 1);
    const auto third = make_u2_weight(1, 2);
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 128});
    const auto shared = placeholder(first, "blk.0.attn_q.weight");
    const auto q = std::make_shared<ov::op::v0::MatMul>(input, shared, false, true);
    const auto tied = std::make_shared<ov::op::v0::MatMul>(input, shared, false, true);
    const auto k = std::make_shared<ov::op::v0::MatMul>(input, placeholder(second, "blk.0.attn_k.weight"), false, true);
    const auto v = std::make_shared<ov::op::v0::MatMul>(input, placeholder(third, "blk.0.attn_v.weight"), false, true);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{q, k, v, tied}, ov::ParameterVector{input});
    materialize(model);
    EXPECT_EQ(count_nodes<ov::op::v0::MatMul>(model), 1);
    const auto x = input_tensor({2, 3, 128});
    const auto outputs = evaluate(model, {x});
    ASSERT_EQ(outputs.size(), 4);
    EXPECT_EQ(outputs[0].get_shape(), (ov::Shape{2, 3, 2}));
    EXPECT_EQ(outputs[1].get_shape(), (ov::Shape{2, 3, 3}));
    EXPECT_EQ(outputs[2].get_shape(), (ov::Shape{2, 3, 1}));
    expect_values(outputs[0], expected_projection(x, first));
    expect_values(outputs[1], expected_projection(x, second));
    expect_values(outputs[2], expected_projection(x, third));
    expect_values(outputs[3], expected_projection(x, first));
    EXPECT_FALSE(is_materialized(first.tensors.weight));
    EXPECT_FALSE(is_materialized(second.tensors.weight));
    EXPECT_FALSE(is_materialized(third.tensors.weight));
}

TEST(GGUFMaterializeWeights, QuantTypeAndConversionPolicyDifferencesPreventFusion) {
    for (const bool different_type : {false, true}) {
        SCOPED_TRACE(different_type);
        const auto first = make_u2_weight(2, 0);
        const auto second = make_u2_weight(3, 1, false, different_type ? 128 : 64);
        const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 128});
        const auto q =
            std::make_shared<ov::op::v0::MatMul>(input, placeholder(first, "blk.0.attn_q.weight"), false, true);
        const auto k = std::make_shared<ov::op::v0::MatMul>(
            input,
            placeholder(second,
                        different_type ? "blk.0.attn_k.weight" : "token_embd.weight",
                        different_type ? GGUF_TYPE_PQ2_0 : GGUF_TYPE_Q2_0),
            false,
            true);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{q, k}, ov::ParameterVector{input});
        materialize(model);
        EXPECT_EQ(count_nodes<ov::op::v0::MatMul>(model), 2);
        const auto x = input_tensor({2, 128});
        const auto outputs = evaluate(model, {x});
        expect_values(outputs[0], expected_projection(x, first));
        expect_values(outputs[1], expected_projection(x, second));
        EXPECT_EQ(*first.repacks, 1);
        EXPECT_EQ(*second.repacks, 1);
    }
}

TEST(GGUFMaterializeWeights, CodeThreeKeepsGroupedZeroPointsWithAndWithoutFusion) {
    for (const bool deferred : {false, true}) {
        SCOPED_TRACE(deferred);
        const auto first = make_u2_weight(2, 0);
        const auto second = make_u2_weight(3, 1, true);
        const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 128});
        const std::shared_ptr<ov::Node> first_node =
            deferred ? placeholder(first, "blk.0.attn_q.weight") : make_weight_node(first.tensors, GGUF_TYPE_Q2_0);
        const std::shared_ptr<ov::Node> second_node =
            deferred ? placeholder(second, "blk.0.attn_k.weight") : make_weight_node(second.tensors, GGUF_TYPE_Q2_0);
        const auto q = std::make_shared<ov::op::v0::MatMul>(input, first_node, false, true);
        const auto k = std::make_shared<ov::op::v0::MatMul>(input, second_node, false, true);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{q, k}, ov::ParameterVector{input});
        if (deferred) {
            materialize(model);
        }
        normalize_u2_zero_points(model);
        size_t grouped = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(node)) {
                const auto shape = subtract->get_input_shape(1);
                ASSERT_EQ(shape.size(), 3);
                EXPECT_EQ(shape[1], 2);
                EXPECT_EQ(shape[2], 1);
                EXPECT_EQ(subtract->get_rt_info().count("gguf_u2_zero_point_shape"), 0);
                ++grouped;
            }
        }
        EXPECT_EQ(grouped, deferred ? 1 : 2);
        const auto x = input_tensor({2, 128});
        const auto outputs = evaluate(model, {x});
        expect_values(outputs[0], expected_projection(x, first));
        expect_values(outputs[1], expected_projection(x, second));
    }
}

class GGUFDeferredNativeLoader : public ::testing::Test {
protected:
    void SetUp() override {
        path = ov::test::utils::generateTestFilePrefix() + ".gguf";
    }

    void TearDown() override {
        if (std::filesystem::exists(path)) {
            ASSERT_TRUE(std::filesystem::is_regular_file(path)) << path;
            EXPECT_TRUE(std::filesystem::remove(path)) << path;
        }
    }

    std::string path;
};

TEST_F(GGUFDeferredNativeLoader, MetadataStaysLazyAndMappedBytesSurviveLoaderRelease) {
    constexpr size_t rows = 2;
    constexpr size_t cols = 64;
    std::vector<uint8_t> bytes(rows * 18, 0);
    std::vector<float> expected(rows * cols);
    for (size_t row = 0; row < rows; ++row) {
        const auto scale = ov::float16(static_cast<float>(row + 1) * 0.5f);
        const auto bits = scale.to_bits();
        std::memcpy(bytes.data() + row * 18, &bits, sizeof(bits));
        for (size_t col = 0; col < cols; ++col) {
            const uint8_t code = (row + col) % 4;
            bytes[row * 18 + 2 + col / 4] |= code << (2 * (col % 4));
            expected[row * cols + col] = (static_cast<int>(code) - 1) * static_cast<float>(scale);
        }
    }
    ov_gguf_test::GgufWriter writer;
    writer.raw_tensor("blk.0.attn_q.weight", {cols, rows}, GGUF_TYPE_Q2_0, bytes);
    ASSERT_TRUE(writer.write(path));
    WeightTensors tensors;
    {
        const auto loaded = get_gguf_data(path);
        const auto& arrays = std::get<1>(loaded);
        tensors = {arrays.at("blk.0.attn_q.weight"), arrays.at("blk.0.attn_q.scales"), arrays.at("blk.0.attn_q.zp")};
        EXPECT_EQ(std::get<2>(loaded).at("blk.0.attn_q.qtype"), GGUF_TYPE_Q2_0);
        EXPECT_EQ(tensors.weight.get_shape(), (ov::Shape{rows, cols}));
        EXPECT_EQ(tensors.scales.get_shape(), (ov::Shape{rows, 1}));
        EXPECT_EQ(tensors.zero_point.get_element_type(), ov::element::u8);
        EXPECT_TRUE(is_deferred(tensors));
        EXPECT_FALSE(is_materialized(tensors.weight));
        EXPECT_FALSE(is_materialized(tensors.scales));
        EXPECT_FALSE(is_materialized(tensors.zero_point));
    }
    const auto node = std::make_shared<GGUFConstant>(tensors, GGUF_TYPE_Q2_0, "blk.0.attn_q.weight");
    tensors = {};
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    materialize(model);
    expect_values(evaluate(model)[0], expected);
}

}  // namespace
