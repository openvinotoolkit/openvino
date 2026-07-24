// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "impls/ocl/kernel_selector_helper.h"
#include "intel_gpu/runtime/layout.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

// Unit coverage for fold_higher_rank_fused_peer(), the function that guards and computes the higher-rank
// fused-eltwise-peer fold in canonicalize_fused_shapes(). The fold is only valid when the peer is an
// order-preserving reshape of its own axes down to the host rank (contiguous, unpadded, planar, default
// format) whose result is broadcast-compatible with the host output. It returns the folded host-rank
// shape on success and std::nullopt otherwise, in which case canonicalize_fused_shapes() falls back to
// the pre-existing rank-extension path (never an assertion). These cases prove that:
//   * the proven equal-total planar reshape (df1) folds to exactly the host shape (kept fused);
//   * a legal higher-rank broadcast peer folds to its broadcast shape (kept fused, distinct from the
//     equal-total case);
//   * every unsafe/unprovable higher-rank shape is rejected (returns nullopt -> safe fallback);
//   * overflowing element counts cannot authorize a fold.

namespace {

layout make_layout(const ov::PartialShape& shape, format fmt, const padding& pad = padding()) {
    return layout(shape, data_types::f16, fmt, pad);
}

}  // namespace

// The proven df1 / reproducer_v3 case: a 5D bfzyx peer folds onto a 4D bfyx host of equal element count.
// The folded shape equals the host shape exactly (equal-total reshape).
TEST(canonicalize_fused_peer_fold, planar_5d_peer_equal_total_folds_to_host) {
    auto peer = make_layout({1, 2, 8, 6, 10}, format::bfzyx);  // 960 elements
    auto host = make_layout({1, 2, 48, 10}, format::bfyx);     // 960 elements (48 = 8 * 6)
    auto folded = fold_higher_rank_fused_peer(peer, host);
    ASSERT_TRUE(folded.has_value());
    EXPECT_EQ(folded->to_shape(), (ov::Shape{1, 2, 48, 10}));

    // The df1 shapes as well.
    auto peer_df1 = make_layout({1, 2, 96, 270, 270}, format::bfzyx);
    auto host_df1 = make_layout({1, 2, 25920, 270}, format::bfyx);  // 25920 = 96 * 270
    auto folded_df1 = fold_higher_rank_fused_peer(peer_df1, host_df1);
    ASSERT_TRUE(folded_df1.has_value());
    EXPECT_EQ(folded_df1->to_shape(), (ov::Shape{1, 2, 25920, 270}));
}

// A 6D peer folding onto a 4D host (still contiguous planar, equal total) is valid; the outermost three
// spatial axes (w,z,y) fold into the host's single first spatial axis.
TEST(canonicalize_fused_peer_fold, planar_6d_peer_equal_total_folds_to_host) {
    auto peer = make_layout({1, 2, 3, 4, 5, 6}, format::bfwzyx);  // 720 elements
    auto host = make_layout({1, 2, 60, 6}, format::bfyx);         // 720 elements (60 = 3 * 4 * 5)
    auto folded = fold_higher_rank_fused_peer(peer, host);
    ASSERT_TRUE(folded.has_value());
    EXPECT_EQ(folded->to_shape(), (ov::Shape{1, 2, 60, 6}));
}

// A legal NumPy broadcast peer (coordinator's case): the peer has FEWER elements than the host and
// broadcasts over the feature dim. It must fold to its own regrouped shape [1,1,48,10] (480 elements),
// which is broadcast-compatible with host [1,2,48,10] (960 elements) -- NOT to the host shape, and it
// must remain distinct from the equal-total reshape case.
TEST(canonicalize_fused_peer_fold, planar_5d_broadcast_peer_folds_to_broadcast_shape) {
    auto peer = make_layout({1, 1, 8, 6, 10}, format::bfzyx);  // 480 elements
    auto host = make_layout({1, 2, 48, 10}, format::bfyx);     // 960 elements
    auto folded = fold_higher_rank_fused_peer(peer, host);
    ASSERT_TRUE(folded.has_value());
    EXPECT_EQ(folded->to_shape(), (ov::Shape{1, 1, 48, 10}));
}

// A peer of the SAME rank as the host is not a higher-rank case; the existing extend path owns it.
TEST(canonicalize_fused_peer_fold, equal_rank_rejected) {
    auto peer = make_layout({1, 2, 48, 10}, format::bfyx);
    auto host = make_layout({1, 2, 48, 10}, format::bfyx);
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// A peer of LOWER rank than the host must not fold (right-aligned extension owns it).
TEST(canonicalize_fused_peer_fold, lower_rank_rejected) {
    auto peer = make_layout({48, 10}, format::bfyx);
    auto host = make_layout({1, 2, 48, 10}, format::bfyx);
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// Higher rank, unequal total, and NOT a legal broadcast (folded first spatial 48 != host 48 is fine but
// the folded feature dim 2 vs host 3 differs and is not 1): must be rejected.
TEST(canonicalize_fused_peer_fold, higher_rank_incompatible_broadcast_rejected) {
    auto peer = make_layout({1, 2, 8, 6, 10}, format::bfzyx);  // folds to [1,2,48,10]
    auto host = make_layout({1, 3, 48, 10}, format::bfyx);     // feature 2 vs 3, neither is 1
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// Higher rank whose folded first spatial axis mismatches the host and is not 1: rejected.
TEST(canonicalize_fused_peer_fold, higher_rank_spatial_mismatch_rejected) {
    auto peer = make_layout({1, 2, 8, 6, 10}, format::bfzyx);  // folds to [1,2,48,10]
    auto host = make_layout({1, 2, 50, 10}, format::bfyx);     // 48 vs 50, neither is 1
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// Padding on either layout makes the physical layout non-contiguous, so folding is unsafe.
TEST(canonicalize_fused_peer_fold, padded_peer_rejected) {
    auto peer = make_layout({1, 2, 8, 6, 10}, format::bfzyx, padding({0, 0, 0, 1, 0}, {0, 0, 0, 1, 0}));
    auto host = make_layout({1, 2, 48, 10}, format::bfyx);
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

TEST(canonicalize_fused_peer_fold, padded_host_rejected) {
    auto peer = make_layout({1, 2, 8, 6, 10}, format::bfzyx);
    auto host = make_layout({1, 2, 48, 10}, format::bfyx, padding({0, 0, 1, 0}, {0, 0, 1, 0}));
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// A blocked peer format is not a plain row-major layout, so regrouping axes would change offsets.
TEST(canonicalize_fused_peer_fold, blocked_peer_rejected) {
    auto peer = make_layout({1, 32, 8, 6, 10}, format::b_fs_zyx_fsv16);
    auto host = make_layout({1, 32, 48, 10}, format::bfyx);
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// A blocked host format is likewise rejected.
TEST(canonicalize_fused_peer_fold, blocked_host_rejected) {
    auto peer = make_layout({1, 32, 8, 6, 10}, format::bfzyx);
    auto host = make_layout({1, 32, 48, 10}, format::b_fs_yx_fsv16);
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// A higher-rank peer whose host is a non-default planar format (byxf) does not describe the same
// physical order after adjust_to_rank, so the fold is rejected.
TEST(canonicalize_fused_peer_fold, non_default_host_rejected) {
    auto peer = make_layout({1, 2, 8, 6, 10}, format::bfzyx);  // 5D default
    auto host = make_layout({1, 2, 48, 10}, format::byxf);     // 4D non-default, equal total
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// Dynamic shapes cannot be proven equal-total and must be rejected.
TEST(canonicalize_fused_peer_fold, dynamic_peer_rejected) {
    auto peer = make_layout(ov::PartialShape{1, 2, 8, 6, ov::Dimension::dynamic()}, format::bfzyx);
    auto host = make_layout({1, 2, 48, 10}, format::bfyx);
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// A host rank below 3 (no batch+feature+spatial to fold into) is rejected.
TEST(canonicalize_fused_peer_fold, host_rank_below_three_rejected) {
    auto peer = make_layout({2, 8, 6}, format::bfyx);
    auto host = make_layout({16, 6}, format::bfyx);
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}

// Overflow guard: a static peer whose folded spatial product overflows size_t must NOT authorize a fold
// through wrapped equality. Using near-2^64 spatial extents makes the true product overflow; the
// overflow-safe computation must reject the fold rather than fold on a wrapped value.
TEST(canonicalize_fused_peer_fold, overflow_rejected) {
    const int64_t big = static_cast<int64_t>(3037000500LL);  // ~2^31.5; big*big*big overflows size_t
    auto peer = make_layout({1, 1, big, big, big}, format::bfzyx);
    // Host chosen so that adjust_to_rank/format checks pass; equality never reached because overflow
    // rejects first. The folded first spatial would be big*big*big (overflow).
    auto host = make_layout({1, 1, big, big}, format::bfyx);
    ASSERT_FALSE(fold_higher_rank_fused_peer(peer, host).has_value());
}
