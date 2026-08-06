// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/decompositions/paged_attention_decomposition.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/paged_attention_onnx.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using ov::op::internal::PagedAttention;

// Graph-level tests for PagedAttentionDecomposition: the internal PagedAttention op is replaced by a
// ScaledDotProductAttention-based subgraph (or a manual MatMul/Softmax core when softcap > 0). The new K/V
// are written into the paged cache with ScatterNDUpdate, and the two cache outputs are always produced.
// Numerical parity against the ONNX Runtime attention_ref oracle is covered by the end-to-end tests in
// onnx_import_com_microsoft.
namespace {

constexpr int64_t NUM_HEADS = 2;
constexpr int64_t KV_NUM_HEADS = 1;
constexpr int64_t HEAD_SIZE = 8;
constexpr int64_t BLOCK_SIZE = 4;
constexpr int64_t NUM_BLOCKS = 4;
constexpr int64_t MAX_BLOCKS = 2;

// Chainable params so each case reads as a one-liner (no C++20 designated initializers - the repo is C++17).
struct PaParams {
    std::string name;
    Dimension num_tokens = 1;
    bool packed = false;
    int64_t local_window_size = -1;
    float softcap = 0.0f;
    bool do_rotary = false;
    bool rotary_interleaved = false;
    Dimension batch = 1;  // past_seqlens length; 1 (static) -> single-sequence path, else varlen path
    element::Type type = element::f32;
    // Expected decomposed structure.
    bool expects_sdpa = true;  // softcap > 0 uses the manual core instead of SDPA

    explicit PaParams(std::string n) : name(std::move(n)) {}
    PaParams& etype(const element::Type& t) {
        type = t;
        return *this;
    }
    PaParams& tokens(const Dimension& t) {
        num_tokens = t;
        return *this;
    }
    PaParams& batch_size(const Dimension& b) {
        batch = b;
        return *this;
    }
    PaParams& pack() {
        packed = true;
        return *this;
    }
    PaParams& window(int64_t w) {
        local_window_size = w;
        return *this;
    }
    PaParams& cap(float c) {
        softcap = c;
        expects_sdpa = false;
        return *this;
    }
    PaParams& rotary(bool interleaved = false) {
        do_rotary = true;
        rotary_interleaved = interleaved;
        return *this;
    }
};

std::shared_ptr<Model> make_pa_model(const PaParams& p) {
    const auto ft = p.type;  // activation / cache float type
    OutputVector args;
    ParameterVector params;
    auto add = [&](const element::Type& et, const PartialShape& ps) {
        auto prm = std::make_shared<op::v0::Parameter>(et, ps);
        args.push_back(prm);
        params.push_back(prm);
    };

    // The internal op receives separate 2-D Q/K/V (the ONNX FE splits the packed QKV before creating it).
    const int64_t q_hidden = (p.packed ? (NUM_HEADS + 2 * KV_NUM_HEADS) : NUM_HEADS) * HEAD_SIZE;
    add(ft, PartialShape{p.num_tokens, q_hidden});                           // 0: query
    add(ft, PartialShape{p.num_tokens, KV_NUM_HEADS * HEAD_SIZE});           // 1: key
    add(ft, PartialShape{p.num_tokens, KV_NUM_HEADS * HEAD_SIZE});           // 2: value
    add(ft, PartialShape{NUM_BLOCKS, BLOCK_SIZE, KV_NUM_HEADS, HEAD_SIZE});  // 3: key_cache
    add(ft, PartialShape{NUM_BLOCKS, BLOCK_SIZE, KV_NUM_HEADS, HEAD_SIZE});  // 4: value_cache
    add(element::i32, PartialShape{p.batch + 1});                            // 5: cumulative_sequence_length
    add(element::i32, PartialShape{p.batch});                                // 6: past_seqlens
    add(element::i32, PartialShape{p.batch, MAX_BLOCKS});                    // 7: block_table
    if (p.do_rotary) {
        add(ft, PartialShape{-1, HEAD_SIZE / 2});  // 8: cos_cache
        add(ft, PartialShape{-1, HEAD_SIZE / 2});  // 9: sin_cache
    }

    const auto pa = std::make_shared<PagedAttention>(args,
                                                     NUM_HEADS,
                                                     KV_NUM_HEADS,
                                                     /*scale*/ 0.0f,
                                                     p.softcap,
                                                     p.local_window_size,
                                                     p.do_rotary,
                                                     p.rotary_interleaved);
    ResultVector results;
    for (size_t i = 0; i < pa->get_output_size(); ++i)
        results.push_back(std::make_shared<op::v0::Result>(pa->output(i)));
    return std::make_shared<Model>(results, params);
}

// Minimal 8-input PagedAttention op for exercising the op's own validation (validate_and_infer_types runs at
// construction). batch is the past_seqlens length (dim 0).
std::shared_ptr<PagedAttention> make_pa_op(const Dimension& batch,
                                           int64_t num_heads,
                                           int64_t kv_num_heads,
                                           int64_t local_window_size) {
    const auto f32 = element::f32;
    OutputVector args{
        std::make_shared<op::v0::Parameter>(f32, PartialShape{-1, num_heads * HEAD_SIZE}),
        std::make_shared<op::v0::Parameter>(f32, PartialShape{-1, kv_num_heads * HEAD_SIZE}),
        std::make_shared<op::v0::Parameter>(f32, PartialShape{-1, kv_num_heads * HEAD_SIZE}),
        std::make_shared<op::v0::Parameter>(f32, PartialShape{NUM_BLOCKS, BLOCK_SIZE, kv_num_heads, HEAD_SIZE}),
        std::make_shared<op::v0::Parameter>(f32, PartialShape{NUM_BLOCKS, BLOCK_SIZE, kv_num_heads, HEAD_SIZE}),
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{batch + 1}),
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{batch}),
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{batch, MAX_BLOCKS}),
    };
    return std::make_shared<PagedAttention>(args,
                                            num_heads,
                                            kv_num_heads,
                                            /*scale*/ 0.0f,
                                            /*softcap*/ 0.0f,
                                            local_window_size,
                                            /*do_rotary*/ false,
                                            /*rotary_interleaved*/ false);
}

}  // namespace

class PagedAttentionDecompositionTest : public testing::TestWithParam<PaParams> {};

TEST_P(PagedAttentionDecompositionTest, decomposes) {
    const auto& p = GetParam();
    auto model = make_pa_model(p);
    pass::Manager manager;
    manager.register_pass<pass::PagedAttentionDecomposition>();
    manager.run_passes(model);

    // The internal op is always replaced.
    EXPECT_EQ(count_ops_of_type<PagedAttention>(model), 0u);
    // Without softcap the core is one ScaledDotProductAttention; with softcap it is a manual MatMul/Softmax
    // core (no SDPA, two MatMuls: Q@K^T and probs@V).
    if (p.expects_sdpa) {
        EXPECT_EQ(count_ops_of_type<op::v13::ScaledDotProductAttention>(model), 1u);
    } else {
        EXPECT_EQ(count_ops_of_type<op::v13::ScaledDotProductAttention>(model), 0u);
        EXPECT_EQ(count_ops_of_type<op::v8::Softmax>(model), 1u);
        EXPECT_GE(count_ops_of_type<op::v0::MatMul>(model), 2u);
    }
    // The new K/V are written into the paged cache with two ScatterNDUpdates (key + value).
    EXPECT_EQ(count_ops_of_type<op::v3::ScatterNDUpdate>(model), 2u);
    // Three outputs: attention output + key_cache_out + value_cache_out.
    EXPECT_EQ(model->get_results().size(), 3u);
}

INSTANTIATE_TEST_SUITE_P(
    PagedAttentionDecomposition,
    PagedAttentionDecompositionTest,
    testing::Values(PaParams{"decode"},
                    PaParams{"prefill"}.tokens(Dimension::dynamic()),
                    PaParams{"packed"}.pack(),
                    PaParams{"rotary"}.rotary(),
                    PaParams{"rotary_interleaved"}.rotary(/*interleaved*/ true),
                    PaParams{"sliding_window"}.window(2),
                    PaParams{"softcap"}.cap(30.0f),
                    PaParams{"softcap_window"}.cap(30.0f).window(2),
                    PaParams{"varlen_dynamic_batch"}.batch_size(Dimension::dynamic()).tokens(Dimension::dynamic()),
                    PaParams{"varlen_static_batch"}.batch_size(2).tokens(Dimension::dynamic()),
                    PaParams{"varlen_window"}.batch_size(2).tokens(Dimension::dynamic()).window(2),
                    PaParams{"varlen_softcap"}.batch_size(2).tokens(Dimension::dynamic()).cap(30.0f)),
    [](const testing::TestParamInfo<PaParams>& i) {
        return i.param.name;
    });

// --- Op-level validation (validate_and_infer_types) ------------------------------------------------
// These guard the assumptions the decomposition relies on. They reject only what is provably unsupported
// from shapes/attributes alone, and must NOT reject the dynamic-shape configs CPU/GPU legitimately run.

TEST(PagedAttentionOpValidation, rejects_local_window_size_zero) {
    // 0 is an empty attention (every query masks all keys, including its own diagonal) and is not valid.
    OV_EXPECT_THROW(make_pa_op(1, NUM_HEADS, KV_NUM_HEADS, /*window*/ 0),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("local_window_size must be -1"));
}

TEST(PagedAttentionOpValidation, rejects_num_heads_not_divisible_by_kv_num_heads) {
    // num_heads must be a multiple of kv_num_heads (each KV head is shared by a group of query heads).
    OV_EXPECT_THROW(make_pa_op(1, /*num_heads*/ 3, /*kv_num_heads*/ 2, /*window*/ -1),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("divisible"));
}

TEST(PagedAttentionOpValidation, allows_static_batch_greater_than_one) {
    // Any batch size is supported: a static batch > 1 takes the general variable-length decomposition path.
    EXPECT_NO_THROW(make_pa_op(/*batch*/ 2, NUM_HEADS, KV_NUM_HEADS, /*window*/ -1));
}

TEST(PagedAttentionOpValidation, allows_dynamic_batch) {
    // A dynamic batch dimension is supported (variable-length path); it cannot be checked from shapes anyway.
    EXPECT_NO_THROW(make_pa_op(Dimension::dynamic(), NUM_HEADS, KV_NUM_HEADS, /*window*/ -1));
}

// --- Decomposed constant values -------------------------------------------------------------------
// The structural TEST_P above checks topology only; a wrong window boundary would still pass it. This
// asserts the actual sliding-window constant the decomposition emits.

TEST(PagedAttentionValues, sliding_window_boundary_uses_window_size_constant) {
    constexpr int64_t W = 5;
    auto model = make_pa_model(PaParams{"win"}.window(W));
    pass::Manager manager;
    manager.register_pass<pass::PagedAttentionDecomposition>();
    manager.run_passes(model);

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

TEST(PagedAttentionValues, bf16_mask_uses_finite_bfloat16_lowest) {
    // bf16 is a spec activation type (T = float16/bfloat16). The additive mask must use the compute type's
    // finite lowest() (never -inf) so a fully-masked row cannot softmax to NaN, and the magnitude must be the
    // bf16 lowest() so it does not overflow to -inf when narrowed. Assert the bf16 branch emits exactly that.
    auto model = make_pa_model(PaParams{"bf16"}.etype(element::bf16));
    pass::Manager manager;
    manager.register_pass<pass::PagedAttentionDecomposition>();
    manager.run_passes(model);

    // The mask is Select(masked_bool, minus_inf, 0) in bf16. Find the bf16 constant equal to bfloat16 lowest().
    const auto expected = std::numeric_limits<ov::bfloat16>::lowest();
    bool found = false;
    for (const auto& op : model->get_ordered_ops()) {
        auto c = as_type_ptr<op::v0::Constant>(op);
        if (c && c->get_element_type() == element::bf16 && ov::shape_size(c->get_shape()) == 1) {
            if (c->cast_vector<ov::bfloat16>()[0] == expected) {
                found = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found) << "bf16 mask must use std::numeric_limits<ov::bfloat16>::lowest() for masked positions";
}
