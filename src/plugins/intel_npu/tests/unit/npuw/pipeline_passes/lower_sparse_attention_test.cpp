// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/lower_sparse_attention.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/openvino.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v13 = ov::op::v13;
namespace v17 = ov::op::v17;

std::shared_ptr<Model> make_model(size_t query_length, bool causal = true) {
    auto query = v0::Constant::create(element::f32, Shape{1, 1, query_length, 1}, {0.0f});
    auto key = v0::Constant::create(element::f32, Shape{1, 1, 8, 1}, {0.0f});
    auto value = v0::Constant::create(element::f32, Shape{1, 1, 8, 1}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto indices = std::make_shared<v0::Parameter>(element::i32, PartialShape{1, 1, query_length, 2});
    auto counts = std::make_shared<v0::Parameter>(element::i32, PartialShape{1, 1, query_length});
    auto length = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    indices->output(0).set_names({"indices"});
    counts->output(0).set_names({"counts"});
    length->output(0).set_names({"length"});
    auto attention = std::make_shared<v17::SparseAttention>(query, key, value, indices, counts, length, 2, 1, causal);
    attention->set_friendly_name("sparse_attention");
    attention->get_rt_info()[ov::npuw::NPUW_SDPA_MASK_RT_KEY] = ov::npuw::NPUW_SDPA_MASK_CAUSAL;
    return std::make_shared<Model>(OutputVector{attention}, ParameterVector{indices, counts, length});
}

void check_numerics(size_t query_length, int64_t valid_length) {
    auto model = make_model(query_length);
    ASSERT_TRUE(ov::npuw::LowerSparseAttention().run_on_model(model));
    std::shared_ptr<v13::ScaledDotProductAttention> attention;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto match = ov::as_type_ptr<v13::ScaledDotProductAttention>(op)) {
            attention = match;
        }
        EXPECT_FALSE(ov::is_type<v17::SparseAttention>(op));
    }
    ASSERT_NE(attention, nullptr);
    EXPECT_EQ(attention->get_input_partial_shape(3), (PartialShape{query_length, 8}));
    EXPECT_EQ(attention->get_input_element_type(3), element::boolean);
    EXPECT_EQ(attention->get_rt_info().count(ov::npuw::NPUW_SDPA_MASK_RT_KEY), 0);

    Tensor indices(element::i32, Shape{1, 1, query_length, 2});
    Tensor counts(element::i32, Shape{1, 1, query_length});
    Tensor length(element::i64, Shape{1});
    length.data<int64_t>()[0] = valid_length;
    std::vector<float> expected(query_length, 0.0f);
    for (size_t row = 0; row < query_length; ++row) {
        const int64_t position = valid_length - static_cast<int64_t>(query_length) + static_cast<int64_t>(row);
        const int64_t visible = position < 0 ? 0 : (position + 1) / 2;
        const int32_t count = static_cast<int32_t>(std::min<int64_t>(visible, 2));
        counts.data<int32_t>()[row] = count;
        indices.data<int32_t>()[2 * row] = count ? static_cast<int32_t>(visible - 1) : -1;
        indices.data<int32_t>()[2 * row + 1] = count > 1 ? 0 : -1;
        float sum = 0.0f;
        int selected = 0;
        for (int64_t physical = 0; physical < 8; ++physical) {
            const int64_t logical = physical - (8 - valid_length);
            if (position < 0 || logical < 0 || logical >= valid_length || logical > position) {
                continue;
            }
            bool included = logical >= ((position + 1) / 2) * 2;
            for (int32_t i = 0; i < count; ++i) {
                included |= logical / 2 == indices.data<int32_t>()[2 * row + i];
            }
            if (included) {
                sum += static_cast<float>(physical + 1);
                ++selected;
            }
        }
        if (selected) {
            expected[row] = sum / static_cast<float>(selected);
        }
    }

    Core core;
    auto compiled = core.compile_model(model, "CPU", hint::inference_precision(element::f32));
    auto request = compiled.create_infer_request();
    request.set_tensor("indices", indices);
    request.set_tensor("counts", counts);
    request.set_tensor("length", length);
    request.infer();
    const auto result = request.get_output_tensor();
    ASSERT_EQ(result.get_shape(), (Shape{1, 1, query_length, 1}));
    for (size_t row = 0; row < query_length; ++row) {
        EXPECT_NEAR(result.data<const float>()[row], expected[row], 1e-5f) << "row " << row;
    }
}

TEST(LowerSparseAttentionTest, PreservesSelectedBlocksAndIncompleteTail) {
    check_numerics(4, 8);
}

TEST(LowerSparseAttentionTest, PreservesLeftPaddingAndEmptyQueryRows) {
    check_numerics(4, 6);
    check_numerics(4, 2);
}

TEST(LowerSparseAttentionTest, PreservesSingleTokenDecode) {
    check_numerics(1, 8);
}

TEST(LowerSparseAttentionTest, MapsLogicalBlocksToNoncontiguousPhysicalCache) {
    auto model = make_model(1);
    auto physical_mask = std::make_shared<v0::Parameter>(element::i64, PartialShape{1, 8});
    physical_mask->output(0).set_names({"physical_mask"});
    model->add_parameters({physical_mask});
    ASSERT_TRUE(ov::npuw::LowerSparseAttention(physical_mask).run_on_model(model));
    Core core;
    auto compiled = core.compile_model(model, "CPU", hint::inference_precision(element::f32));
    auto request = compiled.create_infer_request();
    Tensor indices(element::i32, Shape{1, 1, 1, 2});
    indices.data<int32_t>()[0] = 0;
    indices.data<int32_t>()[1] = -1;
    Tensor counts(element::i32, Shape{1, 1, 1});
    counts.data<int32_t>()[0] = 1;
    Tensor length(element::i64, Shape{1});
    length.data<int64_t>()[0] = 3;
    Tensor mask(element::i64, Shape{1, 8});
    std::fill_n(mask.data<int64_t>(), 8, 0);
    mask.data<int64_t>()[0] = 1;
    mask.data<int64_t>()[1] = 1;
    mask.data<int64_t>()[7] = 1;
    request.set_tensor("indices", indices);
    request.set_tensor("counts", counts);
    request.set_tensor("length", length);
    request.set_tensor("physical_mask", mask);
    request.infer();
    EXPECT_NEAR(request.get_output_tensor().data<const float>()[0], 11.0f / 3.0f, 1e-5f);
}

TEST(LowerSparseAttentionTest, EmptySelectionAtCompleteBlockReturnsZero) {
    auto model = make_model(1);
    ASSERT_TRUE(ov::npuw::LowerSparseAttention().run_on_model(model));
    Core core;
    auto compiled = core.compile_model(model, "CPU", hint::inference_precision(element::f32));
    auto request = compiled.create_infer_request();
    Tensor indices(element::i32, Shape{1, 1, 1, 2});
    std::fill_n(indices.data<int32_t>(), 2, -1);
    Tensor counts(element::i32, Shape{1, 1, 1});
    counts.data<int32_t>()[0] = 0;
    Tensor length(element::i64, Shape{1});
    length.data<int64_t>()[0] = 2;
    request.set_tensor("indices", indices);
    request.set_tensor("counts", counts);
    request.set_tensor("length", length);
    request.infer();
    EXPECT_FLOAT_EQ(request.get_output_tensor().data<const float>()[0], 0.0f);
}

TEST(LowerSparseAttentionTest, PreservesGroupedQueryAttention) {
    auto query = v0::Constant::create(element::f32, Shape{1, 4, 1, 1}, {0.f});
    auto key = v0::Constant::create(element::f32, Shape{1, 2, 4, 1}, {0.f});
    auto value = v0::Constant::create(element::f32, Shape{1, 2, 4, 1}, {1, 2, 3, 4, 10, 20, 30, 40});
    auto indices = v0::Constant::create(element::i32, Shape{1, 1, 1, 1}, {0});
    auto counts = v0::Constant::create(element::i32, Shape{1, 1, 1}, {1});
    auto length = v0::Constant::create(element::i64, Shape{1}, {4});
    auto sparse = std::make_shared<v17::SparseAttention>(query, key, value, indices, counts, length, 2, 1, true);
    auto model = std::make_shared<Model>(OutputVector{sparse}, ParameterVector{});
    ASSERT_TRUE(ov::npuw::LowerSparseAttention().run_on_model(model));
    Core core;
    auto request = core.compile_model(model, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    request.infer();
    const auto output = request.get_output_tensor();
    ASSERT_EQ(output.get_shape(), (Shape{1, 4, 1, 1}));
    EXPECT_NEAR(output.data<const float>()[0], 1.5f, 1e-5f);
    EXPECT_NEAR(output.data<const float>()[1], 1.5f, 1e-5f);
    EXPECT_NEAR(output.data<const float>()[2], 15.f, 1e-5f);
    EXPECT_NEAR(output.data<const float>()[3], 15.f, 1e-5f);
}

TEST(LowerSparseAttentionTest, GroupedAttentionMatchesNativeSparseOperator) {
    auto query = std::make_shared<v0::Parameter>(element::f32, Shape{1, 4, 2, 4});
    auto key = std::make_shared<v0::Parameter>(element::f32, Shape{1, 2, 8, 4});
    auto value = std::make_shared<v0::Parameter>(element::f32, Shape{1, 2, 8, 4});
    auto indices = v0::Constant::create(element::i32, Shape{1, 1, 2, 2}, {0, 1, 0, 2});
    auto counts = v0::Constant::create(element::i32, Shape{1, 1, 2}, {2, 2});
    auto length = v0::Constant::create(element::i64, Shape{1}, {6});
    auto sparse = std::make_shared<v17::SparseAttention>(query, key, value, indices, counts, length, 2, 1, true);
    auto reference = std::make_shared<Model>(OutputVector{sparse}, ParameterVector{query, key, value});
    auto lowered = reference->clone();
    ASSERT_TRUE(ov::npuw::LowerSparseAttention().run_on_model(lowered));
    Core core;
    auto reference_request =
        core.compile_model(reference, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    auto actual_request =
        core.compile_model(lowered, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    for (size_t port = 0; port < 3; ++port) {
        Tensor input(element::f32, reference->input(port).get_shape());
        for (size_t i = 0; i < input.get_size(); ++i) {
            input.data<float>()[i] = std::sin(static_cast<float>(i + 11 * port) * 0.17f);
        }
        reference_request.set_input_tensor(port, input);
        actual_request.set_input_tensor(port, input);
    }
    reference_request.infer();
    actual_request.infer();
    const auto expected = reference_request.get_output_tensor();
    const auto actual = actual_request.get_output_tensor();
    ASSERT_EQ(expected.get_shape(), actual.get_shape());
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-5f) << "element " << i;
    }
}

TEST(LowerSparseAttentionTest, QwenSizedGroupedAttentionMatchesNativeSparseOperator) {
    constexpr size_t queries = 4;
    constexpr size_t heads = 16;
    constexpr size_t kv_heads = 2;
    constexpr size_t capacity = 64;
    constexpr size_t dim = 256;
    auto query = std::make_shared<v0::Parameter>(element::f32, Shape{1, heads, queries, dim});
    auto key = std::make_shared<v0::Parameter>(element::f32, Shape{1, kv_heads, capacity, dim});
    auto value = std::make_shared<v0::Parameter>(element::f32, Shape{1, kv_heads, capacity, dim});
    auto indices = v0::Constant::create(element::i32, Shape{1, 1, queries, 1}, {-1, -1, -1, 0});
    auto counts = v0::Constant::create(element::i32, Shape{1, 1, queries}, {0, 0, 0, 1});
    auto length = v0::Constant::create(element::i64, Shape{1}, {queries});
    auto sparse = std::make_shared<v17::SparseAttention>(query, key, value, indices, counts, length, 4, 1, true);
    auto reference = std::make_shared<Model>(OutputVector{sparse}, ParameterVector{query, key, value});
    auto lowered = reference->clone();
    ASSERT_TRUE(ov::npuw::LowerSparseAttention().run_on_model(lowered));
    Core core;
    auto reference_request =
        core.compile_model(reference, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    auto actual_request =
        core.compile_model(lowered, "CPU", hint::inference_precision(element::f32)).create_infer_request();
    for (size_t port = 0; port < 3; ++port) {
        Tensor input(element::f32, reference->input(port).get_shape());
        for (size_t i = 0; i < input.get_size(); ++i) {
            input.data<float>()[i] = std::sin(static_cast<float>(i + 11 * port) * 0.017f) * 0.5f;
        }
        reference_request.set_input_tensor(port, input);
        actual_request.set_input_tensor(port, input);
    }
    reference_request.infer();
    actual_request.infer();
    const auto expected = reference_request.get_output_tensor();
    const auto actual = actual_request.get_output_tensor();
    ASSERT_EQ(expected.get_shape(), actual.get_shape());
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-5f) << "element " << i;
    }
}

TEST(LowerSparseAttentionTest, LeavesNonCausalAttentionUnchanged) {
    auto model = make_model(4, false);
    EXPECT_FALSE(ov::npuw::LowerSparseAttention().run_on_model(model));
    EXPECT_TRUE(ov::is_type<v17::SparseAttention>(model->get_results()[0]->input_value(0).get_node_shared_ptr()));
}

}  // namespace
