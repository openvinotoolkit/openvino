// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using ov::op::internal::GroupQueryAttention;

// Graph-level tests for GroupQueryAttentionDecomposition: the internal op is replaced by a
// ScaledDotProductAttention-based decomposition, and each feature emits its expected structure
// (sliding-window cache -> ScatterUpdate; smooth_softmax / head_sink -> SDPA sink input). Numerical
// parity against ONNX Runtime is covered by the end-to-end tests in onnx_import_com_microsoft.
namespace {

constexpr int64_t NUM_HEADS = 2;
constexpr int64_t KV_NUM_HEADS = 1;
constexpr int64_t HEAD_SIZE = 16;

std::shared_ptr<op::v0::Constant> make_absent_optional_input() {
    // Optional ONNX inputs are represented as an empty tensor.
    return op::v0::Constant::create(element::dynamic, Shape{0}, {});
}

// Chainable so each test case reads as a one-liner without C++20 designated initializers.
struct GqaParams {
    std::string name;
    bool do_rotary = false;
    bool rotary_interleaved = false;
    int64_t rotary_dim = HEAD_SIZE;
    int64_t kv_cache_bit_width = 0;
    element::Type kv_type = element::f32;
    int64_t local_window_size = -1;
    bool sliding_window_cache = false;
    bool smooth_softmax = false;
    bool head_sink = false;
    bool causal = true;
    bool attention_bias = false;
    Dimension bias_kv_len = Dimension::dynamic();
    Dimension past_len = Dimension::dynamic();
    Dimension seq_len = 1;
    // Expected decomposed structure.
    size_t expected_sdpa_inputs = 4;  // q, k, v, mask (+scale, +sink when a sink is used)
    bool expects_scatter_update = false;

    explicit GqaParams(std::string n) : name(std::move(n)) {}
    GqaParams& rotary(bool interleaved = false, int64_t dim = HEAD_SIZE) {
        do_rotary = true;
        rotary_interleaved = interleaved;
        rotary_dim = dim;
        return *this;
    }
    GqaParams& quant(int64_t bits, element::Type t) {
        kv_cache_bit_width = bits;
        kv_type = t;
        return *this;
    }
    GqaParams& window(int64_t w) {
        local_window_size = w;
        return *this;
    }
    GqaParams& windowed_roll(int64_t w, const Dimension& capacity) {
        local_window_size = w;
        sliding_window_cache = true;
        past_len = capacity;
        expects_scatter_update = true;
        return *this;
    }
    GqaParams& sink_smooth() {
        smooth_softmax = true;
        expected_sdpa_inputs = 6;
        return *this;
    }
    GqaParams& sink_head() {
        head_sink = true;
        expected_sdpa_inputs = 6;
        return *this;
    }
    GqaParams& bidirectional() {
        causal = false;
        return *this;
    }
    // kv_len lets a case declare the bias narrower than the buffer it will be added against (e.g. a
    // static preallocated cache wider than total_sequence_length), reproducing a kv_len/bias-width mismatch.
    GqaParams& bias(const Dimension& kv_len = Dimension::dynamic()) {
        attention_bias = true;
        bias_kv_len = kv_len;
        return *this;
    }
    GqaParams& shape(const Dimension& seq, const Dimension& past) {
        seq_len = seq;
        past_len = past;
        expects_scatter_update = past.is_static();
        return *this;
    }
};

std::shared_ptr<Model> make_gqa_model(const GqaParams& p) {
    const auto f32 = element::f32;
    const int64_t stored_head = p.kv_cache_bit_width == 4 ? HEAD_SIZE / 2 : HEAD_SIZE;  // 4-bit packs 2/byte
    const auto quant = p.kv_cache_bit_width == 0 ? op::internal::GroupQueryAttentionQuantType::NONE
                                                 : op::internal::GroupQueryAttentionQuantType::PER_TENSOR;

    OutputVector args;
    ParameterVector params;
    auto add = [&](const element::Type& et, const PartialShape& ps) {
        auto prm = std::make_shared<op::v0::Parameter>(et, ps);
        args.push_back(prm);
        params.push_back(prm);
    };
    auto pad_to = [&](size_t idx) {
        while (args.size() < idx)
            args.push_back(make_absent_optional_input());  // absent optional input
    };

    // The internal op receives Q/K/V already transposed to [batch, heads, seq, head_size] (the ONNX FE
    // splits the packed QKV before creating it).
    add(f32, PartialShape{1, NUM_HEADS, p.seq_len, HEAD_SIZE});              // 0: query
    add(f32, PartialShape{1, KV_NUM_HEADS, p.seq_len, HEAD_SIZE});           // 1: key
    add(f32, PartialShape{1, KV_NUM_HEADS, p.seq_len, HEAD_SIZE});           // 2: value
    add(p.kv_type, PartialShape{1, KV_NUM_HEADS, p.past_len, stored_head});  // 3: past_key
    add(p.kv_type, PartialShape{1, KV_NUM_HEADS, p.past_len, stored_head});  // 4: past_value
    add(element::i32, PartialShape{1});                                      // 5: seqlens_k
    add(element::i32, PartialShape{});                                       // 6: total_sequence_length
    if (p.do_rotary) {
        add(f32, PartialShape{1, p.rotary_dim / 2});  // 7: cos_cache
        add(f32, PartialShape{1, p.rotary_dim / 2});  // 8: sin_cache
    }
    if (p.attention_bias) {
        pad_to(10);
        add(f32, PartialShape{1, NUM_HEADS, p.seq_len, p.bias_kv_len});  // 10: attention_bias
    }
    if (p.head_sink) {
        pad_to(11);
        add(f32, PartialShape{NUM_HEADS});  // 11: head_sink
    }
    if (p.kv_cache_bit_width != 0) {
        pad_to(12);
        add(element::f32, PartialShape{-1});  // 12: k_scale
        add(element::f32, PartialShape{-1});  // 13: v_scale
    }

    const auto gqa = std::make_shared<GroupQueryAttention>(args,
                                                           NUM_HEADS,
                                                           KV_NUM_HEADS,
                                                           /*scale*/ 0.0f,
                                                           p.do_rotary,
                                                           p.rotary_interleaved,
                                                           p.kv_cache_bit_width,
                                                           quant,
                                                           quant,
                                                           p.local_window_size,
                                                           p.sliding_window_cache,
                                                           p.smooth_softmax,
                                                           p.causal);
    ResultVector results;
    for (size_t i = 0; i < gqa->get_output_size(); ++i)
        results.push_back(std::make_shared<op::v0::Result>(gqa->output(i)));
    return std::make_shared<Model>(results, params);
}

// Minimal 7-input GroupQueryAttention for exercising the op's own validation (validate_and_infer_types
// runs at construction). Only the validated attributes/shapes vary; the KV cache stays unquantized float.
std::shared_ptr<GroupQueryAttention> make_gqa_op(const Dimension& batch,
                                                 const Dimension& seq,
                                                 const Dimension& past,
                                                 int64_t local_window_size,
                                                 bool sliding_window_cache,
                                                 bool causal = true) {
    const auto f32 = element::f32;
    auto q = std::make_shared<op::v0::Parameter>(f32, PartialShape{batch, NUM_HEADS, seq, HEAD_SIZE});
    auto k = std::make_shared<op::v0::Parameter>(f32, PartialShape{batch, KV_NUM_HEADS, seq, HEAD_SIZE});
    auto v = std::make_shared<op::v0::Parameter>(f32, PartialShape{batch, KV_NUM_HEADS, seq, HEAD_SIZE});
    auto past_key = std::make_shared<op::v0::Parameter>(f32, PartialShape{batch, KV_NUM_HEADS, past, HEAD_SIZE});
    auto past_value = std::make_shared<op::v0::Parameter>(f32, PartialShape{batch, KV_NUM_HEADS, past, HEAD_SIZE});
    auto seqlens_k = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{batch});
    auto total_seq = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    OutputVector args{q, k, v, past_key, past_value, seqlens_k, total_seq};
    return std::make_shared<GroupQueryAttention>(args,
                                                 NUM_HEADS,
                                                 KV_NUM_HEADS,
                                                 /*scale*/ 0.0f,
                                                 /*do_rotary*/ false,
                                                 /*rotary_interleaved*/ false,
                                                 /*kv_cache_bit_width*/ 0,
                                                 op::internal::GroupQueryAttentionQuantType::NONE,
                                                 op::internal::GroupQueryAttentionQuantType::NONE,
                                                 local_window_size,
                                                 sliding_window_cache,
                                                 /*smooth_softmax*/ false,
                                                 causal);
}

}  // namespace

class GroupQueryAttentionDecompositionTest : public testing::TestWithParam<GqaParams> {};

TEST_P(GroupQueryAttentionDecompositionTest, decomposes_to_sdpa) {
    const auto& p = GetParam();
    auto model = make_gqa_model(p);
    pass::Manager manager;
    manager.register_pass<pass::GroupQueryAttentionDecomposition>();
    manager.run_passes(model);

    // The internal op is always replaced by exactly one ScaledDotProductAttention.
    EXPECT_EQ(count_ops_of_type<GroupQueryAttention>(model), 0u);
    EXPECT_EQ(count_ops_of_type<op::v13::ScaledDotProductAttention>(model), 1u);

    // The sink input (smooth_softmax / head_sink) shows up as extra SDPA inputs.
    for (const auto& op : model->get_ordered_ops()) {
        if (auto sdpa = as_type_ptr<op::v13::ScaledDotProductAttention>(op)) {
            EXPECT_EQ(sdpa->get_input_size(), p.expected_sdpa_inputs);
        }
    }

    // The windowed rolling / static-cache paths assemble the present buffer with a ScatterUpdate along the
    // sequence axis (2); other cache paths do not. (Partial rotary also emits an unrelated ScatterUpdate to
    // re-attach pass-through channels along the last (channel) axis - filter cache writes by scatter axis.)
    bool has_cache_scatter_update = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto su = as_type_ptr<op::v3::ScatterUpdate>(op)) {
            auto axis_const = as_type_ptr<op::v0::Constant>(su->get_input_node_shared_ptr(3));
            if (axis_const && axis_const->cast_vector<int64_t>() == std::vector<int64_t>{2}) {
                has_cache_scatter_update = true;
                break;
            }
        }
    }
    EXPECT_EQ(has_cache_scatter_update, p.expects_scatter_update);
}

INSTANTIATE_TEST_SUITE_P(GroupQueryAttentionDecomposition,
                         GroupQueryAttentionDecompositionTest,
                         testing::Values(GqaParams{"causal_decode"},
                                         GqaParams{"prefill"}.shape(4, Dimension::dynamic()),
                                         GqaParams{"rotary"}.rotary(),
                                         GqaParams{"rotary_interleaved"}.rotary(/*interleaved*/ true),
                                         GqaParams{"partial_rotary"}.rotary(/*interleaved*/ false, HEAD_SIZE / 2),
                                         GqaParams{"partial_rotary_interleaved"}.rotary(/*interleaved*/ true,
                                                                                        HEAD_SIZE / 2),
                                         GqaParams{"static_past_scatter"}.shape(1, 8),
                                         GqaParams{"static_past_scatter_bias"}.shape(1, 8).bias(5),
                                         GqaParams{"sliding_window"}.window(2),
                                         GqaParams{"attention_bias"}.bias(),
                                         GqaParams{"sliding_window_bias"}.window(2).bias(),
                                         GqaParams{"smooth_softmax"}.sink_smooth(),
                                         GqaParams{"head_sink"}.sink_head(),
                                         GqaParams{"i8_per_tensor"}.quant(8, element::i8),
                                         GqaParams{"i4_per_tensor"}.quant(4, element::u8),
                                         GqaParams{"f8e4m3_per_tensor"}.quant(8, element::f8e4m3),
                                         GqaParams{"windowed_cache"}.windowed_roll(3, 4),
                                         GqaParams{"bidirectional"}.bidirectional()),
                         [](const testing::TestParamInfo<GqaParams>& i) {
                             return i.param.name;
                         });

// --- Op-level validation (validate_and_infer_types) ------------------------------------------------
// These guard the assumptions the decomposition relies on. They must reject only what is provably
// unsupported from shapes/attributes alone, and must NOT reject the dynamic-shape configurations that
// CPU/GPU legitimately run (a dynamic sequence length or batch cannot be checked and stays enabled).

TEST(GroupQueryAttentionOpValidation, rejects_sliding_window_cache_without_window) {
    // sliding_window_cache requires a real window (local_window_size >= 1).
    OV_EXPECT_THROW(make_gqa_op(1, 1, 8, /*window*/ -1, /*swc*/ true),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("sliding_window_cache requires local_window_size"));
}

TEST(GroupQueryAttentionOpValidation, rejects_local_window_size_zero) {
    // 0 is an empty attention (every query masks all keys) and is not a valid config.
    OV_EXPECT_THROW(make_gqa_op(1, 1, 8, /*window*/ 0, /*swc*/ false),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("local_window_size must be -1"));
}

TEST(GroupQueryAttentionOpValidation, allows_static_multi_token_windowed_cache) {
    // A statically-known sequence_length > 1 with a windowed cache (e.g. a fixed-size prefill/context
    // graph) takes the same staging branch a dynamic sequence_length resolving to the same runtime value
    // would, so it is not rejected: whether a step crosses a window eviction is a runtime property
    // (derived from seqlens_k), not something decidable - or worth gating - from the static shape alone.
    EXPECT_NO_THROW(make_gqa_op(1, /*seq*/ 4, 8, /*window*/ 2, /*swc*/ true));
}

TEST(GroupQueryAttentionOpValidation, rejects_static_batch_greater_than_one) {
    // The decomposition uses a scalar past length and assumes batch == 1.
    OV_EXPECT_THROW(make_gqa_op(/*batch*/ 2, 1, 8, /*window*/ -1, /*swc*/ false),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("batch_size == 1"));
}

TEST(GroupQueryAttentionOpValidation, allows_dynamic_sequence_with_windowed_cache) {
    // A dynamic sequence_length is left enabled: at runtime it is typically decode / fitting prefill,
    // which the decomposition handles correctly. Rejecting it would disable CPU/GPU dynamic shapes.
    EXPECT_NO_THROW(make_gqa_op(1, Dimension::dynamic(), 8, /*window*/ 2, /*swc*/ true));
}

TEST(GroupQueryAttentionOpValidation, allows_dynamic_batch) {
    // A dynamic batch dimension cannot be checked from shapes and stays enabled.
    EXPECT_NO_THROW(make_gqa_op(Dimension::dynamic(), 1, 8, /*window*/ -1, /*swc*/ false));
}

TEST(GroupQueryAttentionOpValidation, rejects_local_window_size_with_causal_false) {
    // causal=0 (bidirectional) is mutually exclusive with a sliding window, matching ONNX Runtime
    // (gqa_attention_base.h: causal_ || local_window_size_ == -1).
    OV_EXPECT_THROW(make_gqa_op(1, 1, 8, /*window*/ 2, /*swc*/ false, /*causal*/ false),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("local_window_size requires causal=1"));
}

TEST(GroupQueryAttentionOpValidation, allows_bidirectional_without_window) {
    // causal=0 with the window disabled (the only combination the FE ever produces) must construct cleanly.
    EXPECT_NO_THROW(make_gqa_op(1, 4, 8, /*window*/ -1, /*swc*/ false, /*causal*/ false));
}

// --- Decomposed constant values -------------------------------------------------------------------
// The structural TEST_P above checks topology only; a wrong window boundary or a wrong sink value would
// still pass it. These assert the actual constants the decomposition emits for those two features.

namespace {
void decompose(const std::shared_ptr<Model>& model) {
    pass::Manager manager;
    manager.register_pass<pass::GroupQueryAttentionDecomposition>();
    manager.run_passes(model);
}

std::shared_ptr<op::v13::ScaledDotProductAttention> find_sdpa(const std::shared_ptr<Model>& model) {
    for (const auto& op : model->get_ordered_ops())
        if (auto sdpa = as_type_ptr<op::v13::ScaledDotProductAttention>(op))
            return sdpa;
    return nullptr;
}
}  // namespace

TEST(GroupQueryAttentionValues, sliding_window_boundary_uses_window_size_constant) {
    constexpr int64_t W = 5;
    auto model = make_gqa_model(GqaParams{"win"}.window(W));
    decompose(model);

    // The sliding-window band is the only GreaterEqual in the mask: too_old = GreaterEqual(distance, window),
    // where window is a scalar constant. A wrong boundary here would silently widen/narrow the window.
    std::shared_ptr<Node> greater_eq;
    for (const auto& op : model->get_ordered_ops())
        if (ov::is_type<op::v1::GreaterEqual>(op))
            greater_eq = op;
    ASSERT_NE(greater_eq, nullptr) << "sliding-window GreaterEqual band not found";
    auto window_const = as_type_ptr<op::v0::Constant>(greater_eq->get_input_node_shared_ptr(1));
    ASSERT_NE(window_const, nullptr) << "window boundary is not a constant";
    EXPECT_EQ(window_const->cast_vector<int64_t>(), std::vector<int64_t>{W});
}

TEST(GroupQueryAttentionValues, static_cache_scatter_index_unclamped) {
    // past_seqlen feeds the scatter index directly, with no Minimum clamp in the chain - bounds checking
    // for an out-of-range past_seqlen is ScatterUpdate's own responsibility, not this decomposition's.
    auto model = make_gqa_model(GqaParams{"static_scatter"}.shape(1, 8));
    decompose(model);

    std::shared_ptr<op::v3::ScatterUpdate> cache_scatter;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto su = as_type_ptr<op::v3::ScatterUpdate>(op)) {
            if (ov::is_type<op::v0::Parameter>(su->get_input_node_shared_ptr(0))) {
                cache_scatter = su;
                break;
            }
        }
    }
    ASSERT_NE(cache_scatter, nullptr) << "static full-length cache ScatterUpdate not found";

    // indices = Range(0, S) + past_seqlen
    auto add = as_type_ptr<op::v1::Add>(cache_scatter->get_input_node_shared_ptr(1));
    ASSERT_NE(add, nullptr);
    for (size_t i = 0; i < add->get_input_size(); ++i)
        EXPECT_EQ(as_type_ptr<op::v1::Minimum>(add->get_input_node_shared_ptr(i)), nullptr)
            << "past_seqlen feeding the static-cache scatter index must no longer be clamped";
}

TEST(GroupQueryAttentionValues, smooth_softmax_sink_is_zero) {
    auto model = make_gqa_model(GqaParams{"smooth"}.sink_smooth());
    decompose(model);

    // smooth_softmax adds a zero logit to the softmax denominator: the SDPA sink input (6th) is a
    // Broadcast of a zero constant. A non-zero value here would bias every row's normalization.
    auto sdpa = find_sdpa(model);
    ASSERT_NE(sdpa, nullptr);
    ASSERT_EQ(sdpa->get_input_size(), 6u);
    auto sink_bcast = as_type_ptr<op::v3::Broadcast>(sdpa->get_input_node_shared_ptr(5));
    ASSERT_NE(sink_bcast, nullptr) << "smooth_softmax sink must be a Broadcast of a zero constant";
    auto zero_const = as_type_ptr<op::v0::Constant>(sink_bcast->get_input_node_shared_ptr(0));
    ASSERT_NE(zero_const, nullptr);
    const auto vals = zero_const->cast_vector<float>();
    ASSERT_EQ(vals.size(), 1u);
    EXPECT_FLOAT_EQ(vals[0], 0.0f);
}

TEST(GroupQueryAttentionValues, head_sink_reshapes_input_to_per_head_sink) {
    auto model = make_gqa_model(GqaParams{"hsink"}.sink_head());
    decompose(model);

    // head_sink provides a per-head logit: the SDPA sink input (6th) is the head_sink input reshaped to
    // the [1, num_heads, 1, 1] layout SDPA expects. A wrong target shape would misroute the per-head values.
    auto sdpa = find_sdpa(model);
    ASSERT_NE(sdpa, nullptr);
    ASSERT_EQ(sdpa->get_input_size(), 6u);
    auto sink_reshape = as_type_ptr<op::v1::Reshape>(sdpa->get_input_node_shared_ptr(5));
    ASSERT_NE(sink_reshape, nullptr) << "head_sink sink must be a Reshape of the head_sink input";
    auto shape_const = as_type_ptr<op::v0::Constant>(sink_reshape->get_input_node_shared_ptr(1));
    ASSERT_NE(shape_const, nullptr);
    EXPECT_EQ(shape_const->cast_vector<int64_t>(), (std::vector<int64_t>{1, -1, 1, 1}));
}

TEST(GroupQueryAttentionValues, bidirectional_mask_has_no_causal_comparison) {
    auto model = make_gqa_model(GqaParams{"bidir"}.bidirectional());
    decompose(model);

    // causal=0 must not emit the query-relative Greater(hori, vert) causal comparison; only the single
    // GreaterEqual against the total-length (past + current) threshold remains, and it does not depend
    // on the query row.
    EXPECT_EQ(count_ops_of_type<op::v1::Greater>(model), 0u);
    EXPECT_EQ(count_ops_of_type<op::v1::GreaterEqual>(model), 1u);
}
