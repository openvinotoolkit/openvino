// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "npuw_transformations/patch_sliding_window_mask.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/openvino.hpp"

namespace {

constexpr int64_t kWindow = 4;
constexpr float kFill = -65504.0f;

std::shared_ptr<ov::op::v0::Constant> i64_scalar(int64_t value) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

std::shared_ptr<ov::Node> expand_mask(const ov::Output<ov::Node>& mask) {
    auto batch = std::make_shared<ov::op::v0::Unsqueeze>(mask, i64_scalar(0));
    return std::make_shared<ov::op::v0::Unsqueeze>(batch, i64_scalar(1));
}

// Distance-based attention masks, as exported for LFM3 (optimum-intel, transformers 5.20) and shared by all
// layers:
//
//   Q = Range(past_kv_len, past_kv_len + seq_len) as [seq, 1], K = Range(0, total_seq) as [1, total_seq]
//   full    = Select(!(K <= Q && key_valid), fill, 0)
//   sliding = Select(!(K <= Q && key_valid), fill, Select(Q - K >= window, fill, 0))
//
// Q is a position in the static KV buffer: past tokens are left-aligned at [0, past_kv_len), the current
// chunk is right-aligned after them. past_kv_len is taken from the shape of the past key.
struct MaskParts {
    ov::ParameterVector params;  // input_ids, attention_mask, past_key, position_ids[, length_mask]
    std::shared_ptr<ov::Node> length_mask;
    std::shared_ptr<ov::Node> past_len;
    std::shared_ptr<ov::Node> mask_len;
    std::shared_ptr<ov::Node> q_col;
    std::shared_ptr<ov::Node> k_row;
    std::shared_ptr<ov::Node> invalid;
    std::shared_ptr<ov::Node> window;
    std::shared_ptr<ov::Node> forget;
};

std::shared_ptr<ov::op::v0::Parameter> make_param(ov::element::Type type,
                                                  const ov::PartialShape& shape,
                                                  const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->output(0).set_names({name});
    return param;
}

// `separate_length_mask` takes the K range and the window from another mask than the model's attention_mask.
MaskParts build_mask_parts(int64_t window = kWindow, bool separate_length_mask = false) {
    MaskParts r;
    auto input_ids = make_param(ov::element::i64, ov::PartialShape{1, -1}, "input_ids");
    auto attention_mask = make_param(ov::element::i64, ov::PartialShape{1, -1}, "attention_mask");
    auto past_key = make_param(ov::element::f32, ov::PartialShape{1, 4, -1, 8}, "past_key_values.0.key");
    auto position_ids = make_param(ov::element::i64, ov::PartialShape{1, -1}, "position_ids");
    r.params = {input_ids, attention_mask, past_key, position_ids};
    r.length_mask = attention_mask;
    if (separate_length_mask) {
        auto length_mask = make_param(ov::element::i64, ov::PartialShape{1, -1}, "length_mask");
        r.params.push_back(length_mask);
        r.length_mask = length_mask;
    }

    auto seq_len = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(input_ids),
                                                        i64_scalar(1),
                                                        i64_scalar(0));
    r.past_len = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(past_key),
                                                      i64_scalar(2),
                                                      i64_scalar(0));
    auto full_len = std::make_shared<ov::op::v1::Add>(r.past_len, seq_len);
    auto q_range = std::make_shared<ov::op::v4::Range>(r.past_len, full_len, i64_scalar(1), ov::element::i64);
    r.q_col = std::make_shared<ov::op::v0::Unsqueeze>(q_range, i64_scalar(1));

    r.mask_len = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(r.length_mask),
                                                      i64_scalar(1),
                                                      i64_scalar(0));
    auto k_range = std::make_shared<ov::op::v4::Range>(i64_scalar(0), r.mask_len, i64_scalar(1), ov::element::i64);
    r.k_row = std::make_shared<ov::op::v0::Unsqueeze>(k_range, i64_scalar(0));

    auto causal = std::make_shared<ov::op::v1::LessEqual>(r.k_row, r.q_col);
    auto key_valid_2d = std::make_shared<ov::op::v0::Unsqueeze>(attention_mask, i64_scalar(1));
    auto key_valid =
        std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Unsqueeze>(key_valid_2d, i64_scalar(2)),
                                              ov::element::boolean);
    auto valid = std::make_shared<ov::op::v13::BitwiseAnd>(expand_mask(causal), key_valid);
    r.invalid = std::make_shared<ov::op::v13::BitwiseNot>(valid);

    auto distance = std::make_shared<ov::op::v1::Subtract>(r.q_col, r.k_row);
    r.window = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1}, {window});
    r.forget = std::make_shared<ov::op::v1::GreaterEqual>(distance, r.window);
    return r;
}

std::shared_ptr<ov::Model> make_mask_model(const MaskParts& parts, const ov::Output<ov::Node>& window_condition) {
    auto fill = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {kFill});
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto window_mask = std::make_shared<ov::op::v1::Select>(window_condition, fill, zero);
    auto sliding_mask = std::make_shared<ov::op::v1::Select>(parts.invalid, fill, window_mask);
    auto full_mask = std::make_shared<ov::op::v1::Select>(parts.invalid, fill, zero);
    auto sliding_result = std::make_shared<ov::op::v0::Result>(sliding_mask);
    sliding_result->set_friendly_name("sliding_mask");
    auto full_result = std::make_shared<ov::op::v0::Result>(full_mask);
    full_result->set_friendly_name("full_mask");
    return std::make_shared<ov::Model>(ov::ResultVector{sliding_result, full_result}, parts.params, "distance_mask");
}

std::shared_ptr<ov::Model> getModel(int64_t window = kWindow, bool separate_length_mask = false) {
    const auto parts = build_mask_parts(window, separate_length_mask);
    return make_mask_model(parts, expand_mask(parts.forget));
}

// The window condition after the patch: past keys measure the distance from the query position among the valid
// tokens, keys of the current chunk keep Q - K, and padded query rows are fully masked.
std::shared_ptr<ov::Model> getModelRef() {
    const auto parts = build_mask_parts();
    const auto& mask = parts.length_mask;
    auto one_1d = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto column = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {-1, 1});
    auto past_len_1d = std::make_shared<ov::op::v1::Reshape>(parts.past_len, one_1d, false);
    auto mask_len_1d = std::make_shared<ov::op::v1::Reshape>(parts.mask_len, one_1d, false);

    auto num_valid = std::make_shared<ov::op::v1::ReduceSum>(mask, one_1d, false);
    auto num_invalid = std::make_shared<ov::op::v1::Subtract>(parts.mask_len, num_valid);
    auto query_cache_col = std::make_shared<ov::op::v1::Subtract>(parts.q_col, num_invalid);
    auto history_distance = std::make_shared<ov::op::v1::Subtract>(query_cache_col, parts.k_row);
    auto forget_history = std::make_shared<ov::op::v1::GreaterEqual>(history_distance, parts.window);
    auto is_history = std::make_shared<ov::op::v1::Less>(parts.k_row, parts.past_len);
    auto mapped_forget = std::make_shared<ov::op::v1::Select>(is_history, forget_history, parts.forget);

    auto query_valid = std::make_shared<ov::op::v8::Slice>(mask, past_len_1d, mask_len_1d, one_1d, one_1d);
    auto query_valid_bool = std::make_shared<ov::op::v0::Convert>(query_valid, ov::element::boolean);
    auto query_valid_col = std::make_shared<ov::op::v1::Reshape>(query_valid_bool, column, false);
    auto query_invalid = std::make_shared<ov::op::v1::LogicalNot>(query_valid_col);
    auto clean_forget = std::make_shared<ov::op::v1::LogicalOr>(mapped_forget, query_invalid);
    return make_mask_model(parts, expand_mask(clean_forget));
}

struct StaticInputs {
    size_t past_capacity;                 // static past KV length
    std::vector<int64_t> attention_mask;  // [past_capacity + seq_len]
    std::vector<int64_t> position_ids;    // [seq_len]
};

// Evaluates {sliding, full} masks, each flattened [seq_len, total_seq].
std::vector<std::vector<float>> evaluate(const std::shared_ptr<ov::Model>& original, const StaticInputs& in) {
    const size_t seq_len = in.position_ids.size();
    const size_t total = in.attention_mask.size();
    ov::Tensor input_ids(ov::element::i64, ov::Shape{1, seq_len});
    std::fill_n(input_ids.data<int64_t>(), seq_len, 0);
    ov::Tensor attention_mask(ov::element::i64, ov::Shape{1, total});
    std::copy(in.attention_mask.begin(), in.attention_mask.end(), attention_mask.data<int64_t>());
    ov::Tensor past_key(ov::element::f32, ov::Shape{1, 4, in.past_capacity, 8});
    ov::Tensor position_ids(ov::element::i64, ov::Shape{1, seq_len});
    std::copy(in.position_ids.begin(), in.position_ids.end(), position_ids.data<int64_t>());

    // Evaluate with static shapes, as NPUW compiles the prefill and generate models.
    auto model = original->clone();
    std::map<ov::Output<ov::Node>, ov::PartialShape> shapes;
    const ov::TensorVector inputs{input_ids, attention_mask, past_key, position_ids};
    const auto params = model->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        shapes[params[i]->output(0)] = inputs.at(i).get_shape();
    }
    model->reshape(shapes);
    // Core has no reference evaluation of the Bitwise ops; on booleans they are the Logical ones.
    for (const auto& op : model->get_ordered_ops()) {
        std::shared_ptr<ov::Node> logical;
        if (ov::is_type<ov::op::v13::BitwiseAnd>(op)) {
            logical = std::make_shared<ov::op::v1::LogicalAnd>(op->input_value(0), op->input_value(1));
        } else if (ov::is_type<ov::op::v13::BitwiseNot>(op)) {
            logical = std::make_shared<ov::op::v1::LogicalNot>(op->input_value(0));
        }
        if (logical) {
            OPENVINO_ASSERT(op->get_output_element_type(0) == ov::element::boolean);
            ov::replace_node(op, logical);
        }
    }

    ov::TensorVector outputs;
    for (const auto& result : model->get_results()) {
        outputs.emplace_back(result->get_element_type(), result->get_output_shape(0));
    }
    OPENVINO_ASSERT(model->evaluate(outputs, inputs));

    std::vector<std::vector<float>> masks;
    for (const auto& out : outputs) {
        OPENVINO_ASSERT(out.get_shape() == (ov::Shape{1, 1, seq_len, total}));
        masks.emplace_back(out.data<const float>(), out.data<const float>() + seq_len * total);
    }
    return masks;
}

// Reference sliding mask in cache coordinates: the history keeps the k-th valid token in slot k, and the
// valid tokens of the current chunk follow it. A key is visible if it is valid, not in the future and within
// the window. Padded query rows see nothing. position_ids are not used: the window does not depend on their
// baseline.
std::vector<float> reference_sliding_mask(const StaticInputs& in) {
    const size_t seq_len = in.position_ids.size();
    const size_t total = in.attention_mask.size();
    int64_t num_invalid = 0;
    for (const auto valid : in.attention_mask) {
        num_invalid += valid == 0 ? 1 : 0;
    }
    std::vector<float> mask(seq_len * total, kFill);
    for (size_t q = 0; q < seq_len; ++q) {
        const size_t q_slot = in.past_capacity + q;
        if (in.attention_mask[q_slot] == 0) {
            continue;
        }
        const int64_t q_pos = static_cast<int64_t>(q_slot) - num_invalid;
        for (size_t k = 0; k < total; ++k) {
            const bool is_past = k < in.past_capacity;
            const int64_t distance = is_past ? q_pos - static_cast<int64_t>(k) : static_cast<int64_t>(q_slot - k);
            const bool visible = in.attention_mask[k] != 0 && k <= q_slot && distance < kWindow;
            mask[q * total + k] = visible ? 0.0f : kFill;
        }
    }
    return mask;
}

// Generate step: 6 past tokens left-aligned in a 12-slot past buffer, one new token at position 6.
StaticInputs generate_step() {
    return {12, {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1}, {6}};
}

// Second prefill chunk: 5 past tokens in an 8-slot past buffer, a 4-token chunk with one left pad.
StaticInputs prefill_chunk() {
    return {8, {1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1}, {0, 5, 6, 7}};
}

// The same steps when the request's positions start at 37 instead of 0.
StaticInputs generate_step_with_position_baseline() {
    return {12, {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1}, {43}};
}

StaticInputs prefill_chunk_with_position_baseline() {
    return {8, {1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1}, {0, 42, 43, 44}};
}

void expect_rows_eq(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    const StaticInputs& in,
                    bool only_valid_rows) {
    const size_t total = in.attention_mask.size();
    for (size_t q = 0; q < in.position_ids.size(); ++q) {
        if (only_valid_rows && in.attention_mask[in.past_capacity + q] == 0) {
            continue;
        }
        for (size_t k = 0; k < total; ++k) {
            EXPECT_EQ(actual[q * total + k], expected[q * total + k]) << "query " << q << ", key " << k;
        }
    }
}

class DistanceSlidingMaskTransformationTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::npuw::PatchSlidingWindowMask>();
    }
};

std::shared_ptr<ov::Model> patched(std::shared_ptr<ov::Model> model) {
    ov::npuw::PatchSlidingWindowMask().run_on_model(model);
    return model;
}

}  // namespace

TEST_F(DistanceSlidingMaskTransformationTest, RewritesOnlyTheWindowCondition) {
    model = getModel();
    model_ref = getModelRef();
}

TEST_F(DistanceSlidingMaskTransformationTest, SecondRunLeavesTheGraphUnchanged) {
    model = getModel();
    model_ref = getModelRef();
    manager.register_pass<ov::npuw::PatchSlidingWindowMask>();
}

// A non-positive window is not a sliding window.
TEST_F(DistanceSlidingMaskTransformationTest, NotAppliedForNonPositiveWindow) {
    model = getModel(/*window=*/0);
}

// The ranges must be built from the model's own attention mask.
TEST_F(DistanceSlidingMaskTransformationTest, NotAppliedForAnotherMask) {
    model = getModel(kWindow, /*separate_length_mask=*/true);
}

// Without the patch, Q is the static past length: every left-aligned past token is out of the window.
TEST(DistanceSlidingMaskTest, UnpatchedMaskDropsHistory) {
    const auto sliding = evaluate(getModel(), generate_step())[0];
    for (size_t k = 0; k < 6; ++k) {
        EXPECT_EQ(sliding[k], kFill) << "key " << k;
    }
}

TEST(DistanceSlidingMaskTest, GenerateStepKeepsHistoryWithinWindow) {
    const auto in = generate_step();
    expect_rows_eq(evaluate(patched(getModel()), in)[0], reference_sliding_mask(in), in, /*only_valid_rows=*/false);
}

TEST(DistanceSlidingMaskTest, PrefillChunkKeepsHistoryWithinWindow) {
    const auto in = prefill_chunk();
    expect_rows_eq(evaluate(patched(getModel()), in)[0], reference_sliding_mask(in), in, /*only_valid_rows=*/false);
}

// The window is measured between cache slots, so a nonzero baseline of the positions changes nothing.
TEST(DistanceSlidingMaskTest, IndependentOfPositionBaseline) {
    for (const auto& in : {generate_step_with_position_baseline(), prefill_chunk_with_position_baseline()}) {
        expect_rows_eq(evaluate(patched(getModel()), in)[0],
                       reference_sliding_mask(in),
                       in,
                       /*only_valid_rows=*/false);
    }
}

// The full attention mask shares the causal/padding Select with the sliding one and must stay as it is.
TEST(DistanceSlidingMaskTest, FullAttentionMaskIsUnchanged) {
    for (const auto& in : {generate_step(), prefill_chunk()}) {
        expect_rows_eq(evaluate(patched(getModel()), in)[1],
                       evaluate(getModel(), in)[1],
                       in,
                       /*only_valid_rows=*/false);
    }
}
