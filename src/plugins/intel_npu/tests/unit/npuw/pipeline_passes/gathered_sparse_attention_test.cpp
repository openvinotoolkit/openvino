// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "npuw_transformations/lower_sparse_attention.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/openvino.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v13 = ov::op::v13;
namespace v17 = ov::op::v17;

std::shared_ptr<v0::Parameter> param(const std::string& name, const element::Type& type, const Shape& shape) {
    auto result = std::make_shared<v0::Parameter>(type, shape);
    result->output(0).set_names({name});
    return result;
}

std::shared_ptr<Model> make_model(size_t queries,
                                  size_t keys,
                                  size_t heads,
                                  size_t kv_heads,
                                  size_t dim,
                                  size_t width,
                                  size_t block_size,
                                  bool with_scale = false,
                                  bool causal = true) {
    auto query = param("query", element::f32, {1, heads, queries, dim});
    auto key = param("key", element::f32, {1, kv_heads, keys, dim});
    auto value = param("value", element::f32, {1, kv_heads, keys, dim});
    auto indices = param("indices", element::i32, {1, 1, queries, width});
    auto counts = param("counts", element::i32, {1, 1, queries});
    auto mask = param("mask", element::i64, {1, keys});
    auto length = std::make_shared<v1::ReduceSum>(mask, v0::Constant::create(element::i64, Shape{1}, {1}), false);
    OutputVector inputs{query, key, value, indices, counts, length};
    if (with_scale) {
        inputs.push_back(v0::Constant::create(element::f32, Shape{}, {0.5f}));
    }
    auto attention = std::make_shared<v17::SparseAttention>(inputs, static_cast<int64_t>(block_size), 1, causal);
    attention->set_friendly_name("qsa");
    return std::make_shared<Model>(OutputVector{attention}, ParameterVector{query, key, value, indices, counts, mask});
}

void lower_model(const std::shared_ptr<Model>& model) {
    auto mask = model->input("mask").get_node_shared_ptr();
    ASSERT_TRUE(ov::npuw::LowerSparseAttentionToGathered(mask).run_on_model(model));
    model->validate_nodes_and_infer_types();
}

TEST(GatheredSparseAttentionTest, HandlesNoncontiguousDecodeAndEmptyRows) {
    constexpr size_t keys = 8;
    constexpr size_t block_size = 4;
    auto model = make_model(1, keys, 4, 2, 1, 1, block_size);
    auto native = model->clone();
    ASSERT_NO_FATAL_FAILURE(lower_model(model));
    Core core;
    auto request = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    auto native_request =
        core.compile_model(native, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    Tensor query(element::f32, Shape{1, 4, 1, 1});
    Tensor key(element::f32, Shape{1, 2, keys, 1});
    Tensor values(element::f32, Shape{1, 2, keys, 1});
    Tensor reference_values(element::f32, Shape{1, 2, keys, 1});
    Tensor indices(element::i32, Shape{1, 1, 1, 1});
    Tensor counts(element::i32, Shape{1, 1, 1});
    Tensor mask(element::i64, Shape{1, keys});
    Tensor reference_mask(element::i64, Shape{1, keys});
    std::fill_n(query.data<float>(), query.get_size(), 0.f);
    std::fill_n(key.data<float>(), key.get_size(), 0.f);
    for (size_t head = 0; head < 2; ++head) {
        for (size_t position = 0; position < keys; ++position) {
            values.data<float>()[head * keys + position] = static_cast<float>((head ? 10 : 1) * (position + 1));
        }
    }
    request.set_tensor("query", query);
    request.set_tensor("key", key);
    request.set_tensor("value", values);
    request.set_tensor("indices", indices);
    request.set_tensor("counts", counts);
    request.set_tensor("mask", mask);
    native_request.set_tensor("query", query);
    native_request.set_tensor("key", key);
    native_request.set_tensor("value", reference_values);
    native_request.set_tensor("indices", indices);
    native_request.set_tensor("counts", counts);
    native_request.set_tensor("mask", reference_mask);
    for (size_t valid = 0; valid <= keys; ++valid) {
        SCOPED_TRACE(valid);
        std::fill_n(mask.data<int64_t>(), keys, 0);
        std::fill_n(reference_mask.data<int64_t>(), keys, 0);
        std::fill_n(reference_values.data<float>(), reference_values.get_size(), 0.f);
        if (valid) {
            std::fill_n(mask.data<int64_t>(), valid - 1, 1);
            mask.data<int64_t>()[keys - 1] = 1;
            std::fill_n(reference_mask.data<int64_t>() + keys - valid, valid, 1);
            for (size_t head = 0; head < 2; ++head) {
                for (size_t i = 0; i < valid; ++i) {
                    const auto physical = i == valid - 1 ? keys - 1 : i;
                    reference_values.data<float>()[head * keys + keys - valid + i] =
                        values.data<const float>()[head * keys + physical];
                }
            }
        }
        counts.data<int32_t>()[0] = valid >= block_size ? 1 : 0;
        indices.data<int32_t>()[0] = valid >= block_size ? 0 : -1;
        request.infer();
        const auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.get_shape(), (Shape{1, 4, 1, 1}));
        for (size_t i = 0; i < actual.get_size(); ++i) {
            EXPECT_TRUE(std::isfinite(actual.data<const float>()[i]));
        }
        if (!valid) {
            for (size_t i = 0; i < actual.get_size(); ++i) {
                EXPECT_FLOAT_EQ(actual.data<const float>()[i], 0.f);
            }
            continue;
        }
        native_request.infer();
        const auto expected = native_request.get_output_tensor();
        for (size_t i = 0; i < actual.get_size(); ++i) {
            EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-5f) << "head " << i;
        }
    }
    const std::vector<size_t> physical_positions{0, 2, 4, 7};
    std::fill_n(mask.data<int64_t>(), keys, 0);
    std::fill_n(reference_mask.data<int64_t>(), keys, 0);
    std::fill_n(reference_values.data<float>(), reference_values.get_size(), 0.f);
    for (size_t i = 0; i < physical_positions.size(); ++i) {
        mask.data<int64_t>()[physical_positions[i]] = 1;
        reference_mask.data<int64_t>()[keys - physical_positions.size() + i] = 1;
        for (size_t head = 0; head < 2; ++head) {
            reference_values.data<float>()[head * keys + keys - physical_positions.size() + i] =
                values.data<const float>()[head * keys + physical_positions[i]];
        }
    }
    counts.data<int32_t>()[0] = 1;
    indices.data<int32_t>()[0] = 0;
    request.infer();
    native_request.infer();
    for (size_t i = 0; i < request.get_output_tensor().get_size(); ++i) {
        EXPECT_NEAR(request.get_output_tensor().data<const float>()[i],
                    native_request.get_output_tensor().data<const float>()[i],
                    1e-5f)
            << "head " << i;
    }
}

TEST(GatheredSparseAttentionTest, PreservesPaddedPrefillAndGqa) {
    constexpr size_t queries = 4;
    constexpr size_t keys = 8;
    constexpr size_t dim = 4;
    auto model = make_model(queries, keys, 4, 2, dim, 2, 2);
    auto native = model->clone();
    ASSERT_NO_FATAL_FAILURE(lower_model(model));
    Core core;
    auto request = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    auto reference = core.compile_model(native, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    Tensor query(element::f32, Shape{1, 4, queries, dim});
    Tensor key(element::f32, Shape{1, 2, keys, dim});
    Tensor value(element::f32, Shape{1, 2, keys, dim});
    Tensor indices(element::i32, Shape{1, 1, queries, 2});
    Tensor counts(element::i32, Shape{1, 1, queries});
    Tensor mask(element::i64, Shape{1, keys});
    for (auto* tensor : {&query, &key, &value}) {
        for (size_t i = 0; i < tensor->get_size(); ++i) {
            tensor->data<float>()[i] = std::sin(static_cast<float>(i + tensor->get_size()) * 0.13f);
        }
    }
    for (auto* infer : {&request, &reference}) {
        infer->set_tensor("query", query);
        infer->set_tensor("key", key);
        infer->set_tensor("value", value);
        infer->set_tensor("indices", indices);
        infer->set_tensor("counts", counts);
        infer->set_tensor("mask", mask);
    }
    for (size_t valid : {size_t{2}, size_t{4}}) {
        SCOPED_TRACE(valid);
        std::fill_n(mask.data<int64_t>(), keys, 0);
        std::fill_n(mask.data<int64_t>() + keys - valid, valid, 1);
        std::fill_n(indices.data<int32_t>(), indices.get_size(), -1);
        for (size_t row = 0; row < queries; ++row) {
            const auto position =
                static_cast<int64_t>(valid) - static_cast<int64_t>(queries) + static_cast<int64_t>(row);
            const auto count = static_cast<int32_t>(std::max<int64_t>(0, (position + 1) / 2));
            counts.data<int32_t>()[row] = count;
            for (int32_t i = 0; i < count; ++i) {
                indices.data<int32_t>()[row * 2 + i] = i;
            }
        }
        reference.infer();
        request.infer();
        const auto expected = reference.get_output_tensor();
        const auto actual = request.get_output_tensor();
        ASSERT_EQ(expected.get_shape(), actual.get_shape());
        for (size_t i = 0; i < actual.get_size(); ++i) {
            EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-5f) << "element " << i;
        }
    }
}

TEST(GatheredSparseAttentionTest, UsesExplicitScale) {
    auto model = make_model(1, 4, 1, 1, 1, 1, 2, true);
    ASSERT_NO_FATAL_FAILURE(lower_model(model));
    Core core;
    auto request = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    Tensor query(element::f32, Shape{1, 1, 1, 1});
    Tensor key(element::f32, Shape{1, 1, 4, 1});
    Tensor value(element::f32, Shape{1, 1, 4, 1});
    Tensor indices(element::i32, Shape{1, 1, 1, 1});
    Tensor counts(element::i32, Shape{1, 1, 1});
    Tensor mask(element::i64, Shape{1, 4});
    query.data<float>()[0] = 1.f;
    const std::vector<float> key_values{0.f, 1.f, 2.f, 3.f};
    const std::vector<float> values{1.f, 3.f, 7.f, 9.f};
    std::copy(key_values.begin(), key_values.end(), key.data<float>());
    std::copy(values.begin(), values.end(), value.data<float>());
    indices.data<int32_t>()[0] = 0;
    counts.data<int32_t>()[0] = 1;
    std::fill_n(mask.data<int64_t>(), 4, 1);
    request.set_tensor("query", query);
    request.set_tensor("key", key);
    request.set_tensor("value", value);
    request.set_tensor("indices", indices);
    request.set_tensor("counts", counts);
    request.set_tensor("mask", mask);
    request.infer();
    const auto weight = std::exp(0.5f);
    EXPECT_NEAR(request.get_output_tensor().data<const float>()[0], (1.f + 3.f * weight) / (1.f + weight), 1e-5f);
}

TEST(GatheredSparseAttentionTest, UsesDifferentSelectedBlocksForEachQuery) {
    constexpr size_t queries = 3;
    constexpr size_t keys = 12;
    constexpr size_t dim = 2;
    auto model = make_model(queries, keys, 4, 2, dim, 2, 2);
    auto native = model->clone();
    ASSERT_NO_FATAL_FAILURE(lower_model(model));
    Core core;
    auto request = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    auto reference = core.compile_model(native, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    Tensor query(element::f32, Shape{1, 4, queries, dim});
    Tensor key(element::f32, Shape{1, 2, keys, dim});
    Tensor value(element::f32, Shape{1, 2, keys, dim});
    Tensor indices(element::i32, Shape{1, 1, queries, 2});
    Tensor counts(element::i32, Shape{1, 1, queries});
    Tensor mask(element::i64, Shape{1, keys});
    for (auto* tensor : {&query, &key, &value}) {
        for (size_t i = 0; i < tensor->get_size(); ++i) {
            tensor->data<float>()[i] = std::sin(static_cast<float>(i + tensor->get_size()) * 0.19f);
        }
    }
    const std::vector<int32_t> selected{0, 3, 1, 4, 2, 5};
    std::copy(selected.begin(), selected.end(), indices.data<int32_t>());
    std::fill_n(counts.data<int32_t>(), queries, 2);
    std::fill_n(mask.data<int64_t>(), keys, 1);
    for (auto* infer : {&request, &reference}) {
        infer->set_tensor("query", query);
        infer->set_tensor("key", key);
        infer->set_tensor("value", value);
        infer->set_tensor("indices", indices);
        infer->set_tensor("counts", counts);
        infer->set_tensor("mask", mask);
        infer->infer();
    }
    const auto expected = reference.get_output_tensor();
    const auto actual = request.get_output_tensor();
    ASSERT_EQ(expected.get_shape(), actual.get_shape());
    for (size_t i = 0; i < actual.get_size(); ++i) {
        EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-5f) << "element " << i;
    }
}

TEST(GatheredSparseAttentionTest, ReducesBothAttentionMatmulDimensions) {
    constexpr int64_t keys = 4096;
    constexpr int64_t compact_keys = 2051;
    auto model = make_model(1, keys, 16, 2, 256, 512, 4);
    ASSERT_NO_FATAL_FAILURE(lower_model(model));
    size_t score_count = 0;
    size_t value_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        EXPECT_FALSE(ov::is_type<v13::ScaledDotProductAttention>(node));
        EXPECT_FALSE(ov::is_type<v17::SparseAttention>(node));
        if (const auto matmul = ov::as_type_ptr<v0::MatMul>(node)) {
            if (matmul->get_friendly_name() == "qsa/sparse_scores") {
                ++score_count;
                EXPECT_EQ(matmul->get_output_partial_shape(0), (PartialShape{1, 2, 1, 8, compact_keys}));
                EXPECT_EQ(matmul->get_input_partial_shape(1), (PartialShape{1, 2, 1, compact_keys, 256}));
            }
            if (ov::is_type<ov::op::v8::Softmax>(matmul->get_input_node_shared_ptr(0))) {
                ++value_count;
                EXPECT_EQ(matmul->get_input_partial_shape(0), (PartialShape{1, 2, 1, 8, compact_keys}));
                EXPECT_EQ(matmul->get_input_partial_shape(1), (PartialShape{1, 2, 1, compact_keys, 256}));
            }
        }
    }
    EXPECT_EQ(score_count, 1u);
    EXPECT_EQ(value_count, 1u);
    EXPECT_LT(compact_keys * 100, keys * 51);
}

TEST(GatheredSparseAttentionTest, RejectsIncompatibleMaskAndNoncausalAttention) {
    auto different_mask_model = make_model(1, 8, 1, 1, 4, 2, 2);
    auto different_mask = param("different_mask", element::i64, {1, 8});
    different_mask_model->add_parameters({different_mask});
    EXPECT_FALSE(ov::npuw::LowerSparseAttentionToGathered(different_mask).run_on_model(different_mask_model));
    EXPECT_TRUE(ov::is_type<v17::SparseAttention>(
        different_mask_model->get_results()[0]->input_value(0).get_node_shared_ptr()));

    auto noncausal = make_model(1, 8, 1, 1, 4, 2, 2, false, false);
    EXPECT_FALSE(ov::npuw::LowerSparseAttentionToGathered(noncausal->input("mask").get_node_shared_ptr())
                     .run_on_model(noncausal));
    EXPECT_TRUE(ov::is_type<v17::SparseAttention>(noncausal->get_results()[0]->input_value(0).get_node_shared_ptr()));
}

TEST(GatheredSparseAttentionTest, RunsLongDecodeOnNpu) {
    if (!std::getenv("NPUW_QSA_TEST_NPU")) {
        GTEST_SKIP() << "Set NPUW_QSA_TEST_NPU to run this hardware test";
    }
    constexpr size_t keys = 4096;
    constexpr size_t valid = 3075;
    constexpr size_t width = 512;
    constexpr size_t dim = 256;
    auto model = make_model(1, keys, 16, 2, dim, width, 4);
    ASSERT_NO_FATAL_FAILURE(lower_model(model));
    Core core;
    const auto devices = core.get_available_devices();
    if (std::none_of(devices.begin(), devices.end(), [](const std::string& device) {
            return device == "NPU" || device.find("NPU.") == 0;
        })) {
        GTEST_SKIP() << "The unit-test process cannot initialize an NPU device";
    }
    auto reference = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    auto actual = core.compile_model(model, "NPU").create_infer_request();
    Tensor query(element::f32, Shape{1, 16, 1, dim});
    Tensor key(element::f32, Shape{1, 2, keys, dim});
    Tensor value(element::f32, Shape{1, 2, keys, dim});
    Tensor indices(element::i32, Shape{1, 1, 1, width});
    Tensor counts(element::i32, Shape{1, 1, 1});
    Tensor mask(element::i64, Shape{1, keys});
    for (auto* tensor : {&query, &key, &value}) {
        for (size_t i = 0; i < tensor->get_size(); ++i) {
            tensor->data<float>()[i] = std::sin(static_cast<float>(i % 997) * 0.017f) * 0.25f;
        }
    }
    for (size_t i = 0; i < width; ++i) {
        indices.data<int32_t>()[i] = static_cast<int32_t>(i + 256);
    }
    counts.data<int32_t>()[0] = static_cast<int32_t>(width);
    std::fill_n(mask.data<int64_t>(), keys, 0);
    std::fill_n(mask.data<int64_t>(), valid - 1, 1);
    mask.data<int64_t>()[keys - 1] = 1;
    for (auto* request : {&reference, &actual}) {
        request->set_tensor("query", query);
        request->set_tensor("key", key);
        request->set_tensor("value", value);
        request->set_tensor("indices", indices);
        request->set_tensor("counts", counts);
        request->set_tensor("mask", mask);
    }
    reference.infer();
    actual.infer();
    const auto expected = reference.get_output_tensor();
    const auto observed = actual.get_output_tensor();
    ASSERT_EQ(expected.get_shape(), observed.get_shape());
    for (size_t i = 0; i < observed.get_size(); ++i) {
        EXPECT_TRUE(std::isfinite(observed.data<const float>()[i]));
        EXPECT_NEAR(observed.data<const float>()[i], expected.data<const float>()[i], 1e-2f) << "element " << i;
    }
}

}  // namespace
