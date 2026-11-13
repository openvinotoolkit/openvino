// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <vector>

#include "npuw_transformations/patch_sliding_window_mask.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
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

// The attention masks of an LFM3 export (optimum-intel, transformers 5.20), shared by all layers:
//
//   Q = Range(past_kv_len, past_kv_len + seq_len) as [seq, 1], K = Range(0, total_seq) as [1, total_seq]
//   full    = Select(!(K <= Q && key_valid), fill, 0)
//   sliding = Select(!(K <= Q && key_valid), fill, Select(Q - K >= window, fill, 0))
//
// Q is a position in the static KV buffer: past tokens are left-aligned at [0, past_kv_len), the current
// chunk is right-aligned after them. past_kv_len is taken from the shape of the past key.
struct Lfm3MaskModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v0::Parameter> input_ids;
    std::shared_ptr<ov::op::v0::Parameter> attention_mask;
    std::shared_ptr<ov::op::v0::Parameter> past_key;
    std::shared_ptr<ov::op::v0::Parameter> position_ids;
};

std::shared_ptr<ov::op::v0::Parameter> make_param(ov::element::Type type,
                                                  const ov::PartialShape& shape,
                                                  const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->output(0).set_names({name});
    return param;
}

Lfm3MaskModel build_lfm3_mask_model(int64_t window = kWindow) {
    Lfm3MaskModel r;
    r.input_ids = make_param(ov::element::i64, ov::PartialShape{1, -1}, "input_ids");
    r.attention_mask = make_param(ov::element::i64, ov::PartialShape{1, -1}, "attention_mask");
    r.past_key = make_param(ov::element::f32, ov::PartialShape{1, 4, -1, 8}, "past_key_values.0.key");
    r.position_ids = make_param(ov::element::i64, ov::PartialShape{1, -1}, "position_ids");

    auto seq_len = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(r.input_ids),
                                                        i64_scalar(1),
                                                        i64_scalar(0));
    auto past_len = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(r.past_key),
                                                         i64_scalar(2),
                                                         i64_scalar(0));
    auto full_len = std::make_shared<ov::op::v1::Add>(past_len, seq_len);
    auto q_range = std::make_shared<ov::op::v4::Range>(past_len, full_len, i64_scalar(1), ov::element::i64);
    auto q_col = std::make_shared<ov::op::v0::Unsqueeze>(q_range, i64_scalar(1));

    auto mask_len = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(r.attention_mask),
                                                         i64_scalar(1),
                                                         i64_scalar(0));
    auto k_range = std::make_shared<ov::op::v4::Range>(i64_scalar(0), mask_len, i64_scalar(1), ov::element::i64);
    auto k_row = std::make_shared<ov::op::v0::Unsqueeze>(k_range, i64_scalar(0));

    auto causal = std::make_shared<ov::op::v1::LessEqual>(k_row, q_col);
    auto key_valid_2d = std::make_shared<ov::op::v0::Unsqueeze>(r.attention_mask, i64_scalar(1));
    auto key_valid =
        std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Unsqueeze>(key_valid_2d, i64_scalar(2)),
                                              ov::element::boolean);
    auto valid = std::make_shared<ov::op::v13::BitwiseAnd>(expand_mask(causal), key_valid);
    auto invalid = std::make_shared<ov::op::v13::BitwiseNot>(valid);

    auto fill = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {kFill});
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});

    auto distance = std::make_shared<ov::op::v1::Subtract>(q_col, k_row);
    auto window_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1}, {window});
    auto forget = std::make_shared<ov::op::v1::GreaterEqual>(distance, window_const);
    auto window_mask = std::make_shared<ov::op::v1::Select>(expand_mask(forget), fill, zero);
    auto sliding_mask = std::make_shared<ov::op::v1::Select>(invalid, fill, window_mask);
    auto full_mask = std::make_shared<ov::op::v1::Select>(invalid, fill, zero);

    auto sliding_result = std::make_shared<ov::op::v0::Result>(sliding_mask);
    sliding_result->set_friendly_name("sliding_mask");
    auto full_result = std::make_shared<ov::op::v0::Result>(full_mask);
    full_result->set_friendly_name("full_mask");
    r.model =
        std::make_shared<ov::Model>(ov::ResultVector{sliding_result, full_result},
                                    ov::ParameterVector{r.input_ids, r.attention_mask, r.past_key, r.position_ids},
                                    "lfm3_mask");
    return r;
}

struct StaticInputs {
    size_t past_capacity;                 // static past KV length
    std::vector<int64_t> attention_mask;  // [past_capacity + seq_len]
    std::vector<int64_t> position_ids;    // [seq_len]
};

// Evaluates {sliding, full} masks, each flattened [seq_len, total_seq].
std::vector<std::vector<float>> evaluate(const Lfm3MaskModel& m, const StaticInputs& in) {
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
    auto model = m.model->clone();
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

// Reference sliding mask in temporal positions: past keys at slot k hold position k, keys of the current
// chunk hold position_ids. A key is visible if it is valid, not in the future and within the window.
// Padded query rows see nothing.
std::vector<float> reference_sliding_mask(const StaticInputs& in) {
    const size_t seq_len = in.position_ids.size();
    const size_t total = in.attention_mask.size();
    std::vector<float> mask(seq_len * total, kFill);
    for (size_t q = 0; q < seq_len; ++q) {
        const size_t q_slot = in.past_capacity + q;
        if (in.attention_mask[q_slot] == 0) {
            continue;
        }
        for (size_t k = 0; k < total; ++k) {
            const bool is_past = k < in.past_capacity;
            const int64_t k_pos = is_past ? static_cast<int64_t>(k) : in.position_ids[k - in.past_capacity];
            const bool visible = in.attention_mask[k] != 0 && k <= q_slot && in.position_ids[q] - k_pos < kWindow;
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

}  // namespace

// Without the patch, Q is the static past length: every left-aligned past token is out of the window.
TEST(Lfm3SlidingMaskTest, UnpatchedMaskDropsHistory) {
    auto m = build_lfm3_mask_model();
    const auto in = generate_step();
    const auto sliding = evaluate(m, in)[0];
    for (size_t k = 0; k < 6; ++k) {
        EXPECT_EQ(sliding[k], kFill) << "key " << k;
    }
}

TEST(Lfm3SlidingMaskTest, PatchFires) {
    auto m = build_lfm3_mask_model();
    EXPECT_TRUE(ov::npuw::PatchSlidingWindowMask().run_on_model(m.model));
    EXPECT_NO_THROW(m.model->validate_nodes_and_infer_types());
}

TEST(Lfm3SlidingMaskTest, GenerateStepUsesTemporalPositionsForHistory) {
    auto m = build_lfm3_mask_model();
    ASSERT_TRUE(ov::npuw::PatchSlidingWindowMask().run_on_model(m.model));
    const auto in = generate_step();
    expect_rows_eq(evaluate(m, in)[0], reference_sliding_mask(in), in, /*only_valid_rows=*/false);
}

TEST(Lfm3SlidingMaskTest, PrefillChunkUsesTemporalPositionsForHistory) {
    auto m = build_lfm3_mask_model();
    ASSERT_TRUE(ov::npuw::PatchSlidingWindowMask().run_on_model(m.model));
    const auto in = prefill_chunk();
    expect_rows_eq(evaluate(m, in)[0], reference_sliding_mask(in), in, /*only_valid_rows=*/false);
}

// The full attention mask shares the causal/padding Select with the sliding one and must stay as it is.
TEST(Lfm3SlidingMaskTest, FullAttentionMaskIsUnchanged) {
    auto original = build_lfm3_mask_model();
    auto patched = build_lfm3_mask_model();
    ASSERT_TRUE(ov::npuw::PatchSlidingWindowMask().run_on_model(patched.model));
    for (const auto& in : {generate_step(), prefill_chunk()}) {
        expect_rows_eq(evaluate(patched, in)[1], evaluate(original, in)[1], in, /*only_valid_rows=*/false);
    }
}

TEST(Lfm3SlidingMaskTest, ReapplyIsNoOp) {
    auto m = build_lfm3_mask_model();
    ASSERT_TRUE(ov::npuw::PatchSlidingWindowMask().run_on_model(m.model));
    const auto ops_after_first = m.model->get_ops().size();
    EXPECT_FALSE(ov::npuw::PatchSlidingWindowMask().run_on_model(m.model));
    EXPECT_EQ(m.model->get_ops().size(), ops_after_first);
}

// A non-positive window is not a sliding window: leave the mask alone.
TEST(Lfm3SlidingMaskTest, IgnoresNonPositiveWindow) {
    auto m = build_lfm3_mask_model(/*window=*/0);
    EXPECT_FALSE(ov::npuw::PatchSlidingWindowMask().run_on_model(m.model));
}
