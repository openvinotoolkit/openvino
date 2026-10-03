// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "partitioning/patterns/opt.hpp"
#include "util.hpp"

// The HostGather* passes replace a vocab Gather with a host-side gather fed directly
// from the ids Parameter. Some models guard the ids with a Minimum(N-1, Maximum(-N, ids))
// clamp in between, which the passes now look through (and fold away, relying on
// util::gather() saturating identically). These tests cover all three shapes of the
// input graph for every affected pass:
//   * no clamp at all       - the pre-change form, must keep matching;
//   * the exact [-N, N-1] clamp - the newly supported form, must match;
//   * a clamp with any other bound - must NOT match, folding it would change results.

namespace {

using namespace ov;

constexpr size_t kVocab = 128;
constexpr size_t kHidden = 2048;  // HostGather* passes require the embedding size >= 2048
constexpr size_t kTokens = 4;

enum class Clamp {
    None,        // ids -> Gather
    Exact,       // ids -> Maximum(-N) -> Minimum(N-1) -> Gather
    MaxOnly,     // ids -> Maximum(-N) -> Gather
    WrongBound,  // ids -> Maximum(-N+1) -> Minimum(N-2) -> Gather
};

Output<Node> apply_clamp(const Output<Node>& ids, Clamp clamp) {
    if (clamp == Clamp::None) {
        return ids;
    }
    const bool exact = (clamp != Clamp::WrongBound);
    const int64_t lo = exact ? -static_cast<int64_t>(kVocab) : -static_cast<int64_t>(kVocab) + 1;
    const int64_t hi = exact ? static_cast<int64_t>(kVocab) - 1 : static_cast<int64_t>(kVocab) - 2;

    auto max = std::make_shared<op::v1::Maximum>(ids, op::v0::Constant::create(element::i64, Shape{}, {lo}));
    if (clamp == Clamp::MaxOnly) {
        return max;
    }
    return std::make_shared<op::v1::Minimum>(max, op::v0::Constant::create(element::i64, Shape{}, {hi}));
}

std::shared_ptr<op::v0::Constant> axis0() {
    return op::v0::Constant::create(element::i32, Shape{}, {0});
}

bool has_gather(const std::shared_ptr<ov::Model>& model) {
    for (auto&& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<op::v8::Gather>(node)) {
            return true;
        }
    }
    return false;
}

// Param(ids) -> [clamp] ->
// Param(W:f16)[V,H] -----> Gather -> Convert(f32) -> Result
std::shared_ptr<ov::Model> make_host_gather_model(Clamp clamp) {
    auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, kTokens});
    auto w = std::make_shared<op::v0::Parameter>(element::f16, Shape{kVocab, kHidden});
    auto g = std::make_shared<op::v8::Gather>(w, apply_clamp(ids, clamp), axis0());
    auto out = std::make_shared<op::v0::Convert>(g, element::f32);
    return std::make_shared<ov::Model>(ResultVector{std::make_shared<op::v0::Result>(out)}, ParameterVector{ids, w});
}

// Param(ids) -> [clamp] -------------> (indices of both Gathers)
// Param(W:i4)[V,H]  -> Convert(f16) -> Gather(W) -.
// Param(S:f16)[V,1] -----------------> Gather(S) -> Multiply -> Result
std::shared_ptr<ov::Model> make_host_gather_dq_model(Clamp clamp) {
    auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, kTokens});
    auto clamped = apply_clamp(ids, clamp);

    auto w = std::make_shared<op::v0::Parameter>(element::i4, Shape{kVocab, kHidden});
    auto cvtw = std::make_shared<op::v0::Convert>(w, element::f16);
    auto s = std::make_shared<op::v0::Parameter>(element::f16, Shape{kVocab, 1});

    auto gw = std::make_shared<op::v8::Gather>(cvtw, clamped, axis0());
    auto gs = std::make_shared<op::v8::Gather>(s, clamped, axis0());
    auto mul = std::make_shared<op::v1::Multiply>(gw, gs);
    return std::make_shared<ov::Model>(ResultVector{std::make_shared<op::v0::Result>(mul)}, ParameterVector{ids, w, s});
}

// Param(ids) -> [clamp] ---> (indices of all three Gathers)
// Param(W:u8)  ------------> Gather(W) -> Convert(f16) -.
// Param(Z:u8)  ------------> Gather(Z) -> Convert(f16) -> Subtract -.
// Param(S:f16) ------------> Gather(S) -----------------------------> Multiply -> Convert(f32) -> Result
std::shared_ptr<ov::Model> make_host_gather_quant_asymm_model(Clamp clamp) {
    auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, kTokens});
    auto clamped = apply_clamp(ids, clamp);

    auto w = std::make_shared<op::v0::Parameter>(element::u8, Shape{kVocab, kHidden});
    auto z = std::make_shared<op::v0::Parameter>(element::u8, Shape{kVocab, 1});
    auto s = std::make_shared<op::v0::Parameter>(element::f16, Shape{kVocab, 1});

    auto gw = std::make_shared<op::v8::Gather>(w, clamped, axis0());
    auto gz = std::make_shared<op::v8::Gather>(z, clamped, axis0());
    auto gs = std::make_shared<op::v8::Gather>(s, clamped, axis0());

    auto cvtw = std::make_shared<op::v0::Convert>(gw, element::f16);
    auto cvtz = std::make_shared<op::v0::Convert>(gz, element::f16);
    auto sub = std::make_shared<op::v1::Subtract>(cvtw, cvtz);
    auto mul = std::make_shared<op::v1::Multiply>(sub, gs);
    auto out = std::make_shared<op::v0::Convert>(mul, element::f32);

    return std::make_shared<ov::Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                       ParameterVector{ids, w, z, s});
}

// Param(ids) -> [clamp] --> (indices of both Gathers)
// Param(W:i4)  -----------> Gather(W) -> Convert(f16) -.
// Param(S:f16) -----------> Gather(S) -----------------> Multiply -> Reshape -> Convert(f32) -> Result
// W is also read by a second (tied-embedding) consumer: the pass requires the weight
// to have more than one reader.
std::shared_ptr<ov::Model> make_host_gather_quant_symm_model(Clamp clamp) {
    auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, kTokens});
    auto clamped = apply_clamp(ids, clamp);

    auto w = std::make_shared<op::v0::Parameter>(element::i4, Shape{kVocab, kHidden});
    auto s = std::make_shared<op::v0::Parameter>(element::f16, Shape{kVocab, 1});

    auto gw = std::make_shared<op::v8::Gather>(w, clamped, axis0());
    auto gs = std::make_shared<op::v8::Gather>(s, clamped, axis0());
    auto cvtw = std::make_shared<op::v0::Convert>(gw, element::f16);
    auto mul = std::make_shared<op::v1::Multiply>(cvtw, gs);

    auto rshp_c =
        op::v0::Constant::create(element::i32,
                                 Shape{3},
                                 std::vector<int32_t>{1, static_cast<int32_t>(kTokens), static_cast<int32_t>(kHidden)});
    auto rshp = std::make_shared<op::v1::Reshape>(mul, rshp_c, false);
    auto out = std::make_shared<op::v0::Convert>(rshp, element::f32);

    auto tied = std::make_shared<op::v0::Convert>(w, element::f16);

    return std::make_shared<ov::Model>(
        ResultVector{std::make_shared<op::v0::Result>(out), std::make_shared<op::v0::Result>(tied)},
        ParameterVector{ids, w, s});
}

// Const(T:f32)[16]  -> Convert(f16) ----.
// Param(W:u4)[V,H]  -> Convert(i32) ----> Gather(codebook) -.
// Param(S:f16)[V,1] -----------------------------------------> Multiply -> Convert(f32) -.
// Param(ids) -> Convert(i64) -> [clamp] --------------------------------> Gather(vocab) -> Result
std::shared_ptr<ov::Model> make_host_gather_cb4_model(Clamp clamp) {
    auto ids = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, kTokens});
    auto cvtids = std::make_shared<op::v0::Convert>(ids, element::i64);
    auto clamped = apply_clamp(cvtids, clamp);

    auto w = std::make_shared<op::v0::Parameter>(element::u4, Shape{kVocab, kHidden});
    auto cvtw = std::make_shared<op::v0::Convert>(w, element::i32);

    auto table = op::v0::Constant::create(element::f32, Shape{16}, std::vector<float>(16, 1.0f));
    auto cvtt = std::make_shared<op::v0::Convert>(table, element::f16);
    auto wg = std::make_shared<op::v8::Gather>(cvtt, cvtw, axis0());

    auto s = std::make_shared<op::v0::Parameter>(element::f16, Shape{kVocab, 1});
    auto mul = std::make_shared<op::v1::Multiply>(wg, s);
    auto cvtmul = std::make_shared<op::v0::Convert>(mul, element::f32);

    auto g = std::make_shared<op::v8::Gather>(cvtmul, clamped, axis0());
    return std::make_shared<ov::Model>(ResultVector{std::make_shared<op::v0::Result>(g)}, ParameterVector{ids, w, s});
}

template <typename Pass, typename... Args>
void run_pass(const std::shared_ptr<ov::Model>& model, Args&&... args) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<Pass>(std::forward<Args>(args)...);
    rewr.run_on_model(model);
}

}  // namespace

using namespace ov;
using namespace ov::npuw::patterns::opt;

// ─────────────────────────────────────────────────────────────────────────────
// HostGather
// ─────────────────────────────────────────────────────────────────────────────
TEST(HostGatherIdClamp, HostGather_NoClamp_Matches) {
    auto model = make_host_gather_model(Clamp::None);

    Context ctx;
    run_pass<HostGather>(model, std::ref(ctx));

    EXPECT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_FALSE(has_gather(model));
}

TEST(HostGatherIdClamp, HostGather_ExactClamp_Matches) {
    auto model = make_host_gather_model(Clamp::Exact);

    Context ctx;
    run_pass<HostGather>(model, std::ref(ctx));

    EXPECT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_FALSE(has_gather(model));
}

TEST(HostGatherIdClamp, HostGather_MaximumOnlyClamp_Matches) {
    auto model = make_host_gather_model(Clamp::MaxOnly);

    Context ctx;
    run_pass<HostGather>(model, std::ref(ctx));

    EXPECT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_FALSE(has_gather(model));
}

TEST(HostGatherIdClamp, HostGather_WrongBoundClamp_DoesNotMatch) {
    auto model = make_host_gather_model(Clamp::WrongBound);

    Context ctx;
    run_pass<HostGather>(model, std::ref(ctx));

    EXPECT_FALSE(ctx.params_to_gather.has_value());
    EXPECT_TRUE(has_gather(model));
}

// ─────────────────────────────────────────────────────────────────────────────
// HostGatherDQ
// ─────────────────────────────────────────────────────────────────────────────
TEST(HostGatherIdClamp, HostGatherDQ_NoClamp_Matches) {
    auto model = make_host_gather_dq_model(Clamp::None);

    Context ctx;
    run_pass<HostGatherDQ>(model, std::ref(ctx));

    EXPECT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_EQ(ctx.params_to_unpack.size(), 1u);
}

TEST(HostGatherIdClamp, HostGatherDQ_ExactClamp_Matches) {
    auto model = make_host_gather_dq_model(Clamp::Exact);

    Context ctx;
    run_pass<HostGatherDQ>(model, std::ref(ctx));

    EXPECT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_EQ(ctx.params_to_unpack.size(), 1u);
}

TEST(HostGatherIdClamp, HostGatherDQ_WrongBoundClamp_DoesNotMatch) {
    auto model = make_host_gather_dq_model(Clamp::WrongBound);

    Context ctx;
    run_pass<HostGatherDQ>(model, std::ref(ctx));

    EXPECT_FALSE(ctx.params_to_gather.has_value());
    EXPECT_TRUE(ctx.params_to_unpack.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// HostGatherQuantAsymm
// ─────────────────────────────────────────────────────────────────────────────
TEST(HostGatherIdClamp, HostGatherQuantAsymm_NoClamp_Matches) {
    auto model = make_host_gather_quant_asymm_model(Clamp::None);

    Context ctx;
    run_pass<HostGatherQuantAsymm<op::v0::Parameter>>(model, std::ref(ctx), false);

    ASSERT_TRUE(ctx.params_to_quant_gather_unpack.has_value());
    EXPECT_EQ(ctx.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.size(), 1u);
}

TEST(HostGatherIdClamp, HostGatherQuantAsymm_ExactClamp_Matches) {
    auto model = make_host_gather_quant_asymm_model(Clamp::Exact);

    Context ctx;
    run_pass<HostGatherQuantAsymm<op::v0::Parameter>>(model, std::ref(ctx), false);

    ASSERT_TRUE(ctx.params_to_quant_gather_unpack.has_value());
    EXPECT_EQ(ctx.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.size(), 1u);
}

TEST(HostGatherIdClamp, HostGatherQuantAsymm_WrongBoundClamp_DoesNotMatch) {
    auto model = make_host_gather_quant_asymm_model(Clamp::WrongBound);

    Context ctx;
    run_pass<HostGatherQuantAsymm<op::v0::Parameter>>(model, std::ref(ctx), false);

    EXPECT_FALSE(ctx.params_to_quant_gather_unpack.has_value());
}

// ─────────────────────────────────────────────────────────────────────────────
// HostGatherQuantSymm
// ─────────────────────────────────────────────────────────────────────────────
TEST(HostGatherIdClamp, HostGatherQuantSymm_NoClamp_Matches) {
    auto model = make_host_gather_quant_symm_model(Clamp::None);

    Context ctx;
    run_pass<HostGatherQuantSymm<op::v0::Parameter>>(model, std::ref(ctx), false);

    ASSERT_TRUE(ctx.params_to_quant_gather_unpack.has_value());
    EXPECT_EQ(ctx.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.size(), 1u);
}

TEST(HostGatherIdClamp, HostGatherQuantSymm_ExactClamp_Matches) {
    auto model = make_host_gather_quant_symm_model(Clamp::Exact);

    Context ctx;
    run_pass<HostGatherQuantSymm<op::v0::Parameter>>(model, std::ref(ctx), false);

    ASSERT_TRUE(ctx.params_to_quant_gather_unpack.has_value());
    EXPECT_EQ(ctx.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.size(), 1u);
}

TEST(HostGatherIdClamp, HostGatherQuantSymm_WrongBoundClamp_DoesNotMatch) {
    auto model = make_host_gather_quant_symm_model(Clamp::WrongBound);

    Context ctx;
    run_pass<HostGatherQuantSymm<op::v0::Parameter>>(model, std::ref(ctx), false);

    EXPECT_FALSE(ctx.params_to_quant_gather_unpack.has_value());
}

// ─────────────────────────────────────────────────────────────────────────────
// HostGatherCB4
// ─────────────────────────────────────────────────────────────────────────────
TEST(HostGatherIdClamp, HostGatherCB4_NoClamp_Matches) {
    auto model = make_host_gather_cb4_model(Clamp::None);

    Context ctx;
    run_pass<HostGatherCB4>(model, std::ref(ctx));

    EXPECT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_EQ(ctx.params_to_nf4_gather.size(), 1u);
}

TEST(HostGatherIdClamp, HostGatherCB4_ExactClamp_Matches) {
    auto model = make_host_gather_cb4_model(Clamp::Exact);

    Context ctx;
    run_pass<HostGatherCB4>(model, std::ref(ctx));

    EXPECT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_EQ(ctx.params_to_nf4_gather.size(), 1u);
}

TEST(HostGatherIdClamp, HostGatherCB4_WrongBoundClamp_DoesNotMatch) {
    auto model = make_host_gather_cb4_model(Clamp::WrongBound);

    Context ctx;
    run_pass<HostGatherCB4>(model, std::ref(ctx));

    EXPECT_FALSE(ctx.params_to_gather.has_value());
    EXPECT_TRUE(ctx.params_to_nf4_gather.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// util::gather - the host-side replacement must reproduce both the folded-away
// clamp (saturation) and Gather's negative-index semantics.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HostGatherIdClamp, UtilGatherSaturatesOutOfRangeIds) {
    constexpr size_t kRows = 4;
    constexpr size_t kCols = 3;

    auto src = ov::Tensor(element::f32, Shape{kRows, kCols});
    auto* psrc = src.data<float>();
    for (size_t r = 0; r < kRows; r++) {
        for (size_t c = 0; c < kCols; c++) {
            psrc[r * kCols + c] = static_cast<float>(r);
        }
    }

    const std::vector<int64_t> ids = {0, 3, -1, -4, 1000, -1000};
    auto idx = ov::Tensor(element::i64, Shape{1, ids.size()});
    std::copy(ids.begin(), ids.end(), idx.data<int64_t>());

    auto dst = ov::Tensor(element::f32, Shape{1, ids.size(), kCols});
    ov::npuw::util::gather(ov::get_tensor_impl(src), ov::get_tensor_impl(idx), ov::get_tensor_impl(dst));

    // 1000 saturates to 3, -1000 saturates to -4 which then wraps to row 0.
    const std::vector<float> expected_rows = {0.f, 3.f, 3.f, 0.f, 3.f, 0.f};
    const auto* pdst = dst.data<float>();
    for (size_t r = 0; r < expected_rows.size(); r++) {
        for (size_t c = 0; c < kCols; c++) {
            EXPECT_EQ(pdst[r * kCols + c], expected_rows[r]) << "row " << r << " col " << c;
        }
    }
}
