// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_ggml_moe_gather_to_gathermatmul.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "transformations/op_conversions/convert_gather_matmul_to_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {

// Build the ggml-openvino frontend's public-op MoE expert-matmul carrier, exactly as
// gemma4_moe_pr_v3's translate_mul_mat_id emits it (the pattern the pass must match):
//
//   W(low-bit)[n_expert, m*k/g, g] -> Convert f16 -> [Subtract(zp)] -> Multiply(scale)
//     -> Reshape([n_expert, m*k]) -> [Convert f32]                 (CompressedWeightsBlock)
//   sel  = Gather(block, ids[n_tokens, n_used], axis=0)            -> [nt, nu, m*k]
//   selr = Reshape(sel, dynamic[nt, nu, m, -1])                    -> [nt, nu, m, k]
//   acts = Param[nt, 1, k] -> Broadcast[nt, nu, k] -> Unsqueeze(2) -> [nt, nu, 1, k]
//   out  = MatMul(acts, selr, transpose_b=true)                    -> [nt, nu, 1, m]
//   out  = Reshape(out, dynamic[1, nt, nu, m])
//
// `w_type` selects u4/u8 (asymmetric, with zp) or i4/i8 (symmetric, no zp).
struct CarrierParams {
    ov::element::Type w_type;
    bool with_zp;
    int64_t n_expert = 8;
    int64_t m = 16;   // per-expert output rows (N)
    int64_t k = 64;   // per-expert input width (K)
    int64_t group = 32;
    int64_t n_used = 8;
};

std::shared_ptr<ov::Node> make_i64(const std::vector<int64_t>& v) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
}

std::shared_ptr<ov::Node> gather_dim(const ov::Output<ov::Node>& shape_of, int64_t idx) {
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    return std::make_shared<ov::op::v8::Gather>(shape_of, make_i64({idx}), axis);
}

std::shared_ptr<ov::Model> build_carrier(const CarrierParams& p) {
    const int64_t E = p.n_expert;
    const int64_t mkg = (p.m * p.k) / p.group;

    // Compressed-weights block (rank-2 [E, m*k]).
    auto w = ov::op::v0::Constant::create(p.w_type, ov::Shape{(size_t)E, (size_t)mkg, (size_t)p.group},
                                          std::vector<int64_t>((size_t)(E * mkg * p.group), 1));
    std::shared_ptr<ov::Node> deq = std::make_shared<ov::op::v0::Convert>(w, ov::element::f16);
    if (p.with_zp) {
        auto zp = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{(size_t)E, (size_t)mkg, 1},
                                               std::vector<float>((size_t)(E * mkg), 0.5f));
        deq = std::make_shared<ov::op::v1::Subtract>(deq, zp);
    }
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{(size_t)E, (size_t)mkg, 1},
                                              std::vector<float>((size_t)(E * mkg), 0.1f));
    deq = std::make_shared<ov::op::v1::Multiply>(deq, scale);
    auto block = std::make_shared<ov::op::v1::Reshape>(deq, make_i64({E, p.m * p.k}), false);
    std::shared_ptr<ov::Node> block_f32 = std::make_shared<ov::op::v0::Convert>(block, ov::element::f32);

    // ids [n_tokens, n_used], activations [n_tokens, 1, k]; n_tokens dynamic.
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, p.n_used});
    auto acts = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, p.k});

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto sel = std::make_shared<ov::op::v8::Gather>(block_f32, ids, gather_axis);  // [nt, nu, m*k]

    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto acts_shape = std::make_shared<ov::op::v3::ShapeOf>(acts, ov::element::i64);

    // selr: [nt, nu, m, -1]
    auto split_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gather_dim(ids_shape, 0), gather_dim(ids_shape, 1), make_i64({p.m}), make_i64({-1})}, 0);
    auto selr = std::make_shared<ov::op::v1::Reshape>(sel, split_dims, false);

    // acts -> broadcast [nt, nu, k] -> unsqueeze(2) -> [nt, nu, 1, k]
    auto bcast_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gather_dim(acts_shape, 0), gather_dim(ids_shape, 1), gather_dim(acts_shape, 2)}, 0);
    auto acts_b = std::make_shared<ov::op::v3::Broadcast>(acts, bcast_dims, ov::op::BroadcastType::BIDIRECTIONAL);
    auto acts_e = std::make_shared<ov::op::v0::Unsqueeze>(acts_b, make_i64({2}));

    auto matmul = std::make_shared<ov::op::v0::MatMul>(acts_e, selr, false, true);  // [nt, nu, 1, m]

    auto out_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{make_i64({1}), gather_dim(ids_shape, 0), gather_dim(ids_shape, 1), make_i64({p.m})}, 0);
    auto out = std::make_shared<ov::op::v1::Reshape>(matmul, out_dims, false);

    return std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{ids, acts});
}

// GatherMatmulCompressed derives from GatherMatmul, so ov::is_type<GatherMatmul> also
// matches the compressed subtype. Count the *plain* (non-compressed) GatherMatmul only.
size_t count_gather_matmul(const std::shared_ptr<ov::Model>& m) {
    size_t count = 0;
    for (const auto& node : m->get_ops()) {
        if (ov::is_type<ov::op::internal::GatherMatmul>(node) &&
            !ov::is_type<ov::op::internal::GatherMatmulCompressed>(node)) {
            ++count;
        }
    }
    return count;
}
size_t count_gather_matmul_compressed(const std::shared_ptr<ov::Model>& m) {
    return count_ops_of_type<ov::op::internal::GatherMatmulCompressed>(m);
}
size_t count_matmul(const std::shared_ptr<ov::Model>& m) {
    return count_ops_of_type<ov::op::v0::MatMul>(m);
}

}  // namespace

// ---- Positive: the pass fires and produces a GatherMatmul, removing the expert MatMul ----

TEST(ConvertGgmlMoeGatherToGatherMatmul, FiresOnU4WithZeroPoint) {
    auto model = build_carrier({ov::element::u4, /*with_zp=*/true});
    ASSERT_EQ(count_matmul(model), 1u);
    ASSERT_EQ(count_gather_matmul(model), 0u);

    ov::pass::Manager manager;
    manager.register_pass<ConvertGgmlMoeGatherToGatherMatmul>();
    manager.run_passes(model);

    // The expert MatMul is rewritten to exactly one GatherMatmul.
    EXPECT_EQ(count_gather_matmul(model), 1u);
    EXPECT_EQ(count_matmul(model), 0u);
}

TEST(ConvertGgmlMoeGatherToGatherMatmul, FiresOnU8WithZeroPoint) {
    auto model = build_carrier({ov::element::u8, /*with_zp=*/true});

    ov::pass::Manager manager;
    manager.register_pass<ConvertGgmlMoeGatherToGatherMatmul>();
    manager.run_passes(model);

    EXPECT_EQ(count_gather_matmul(model), 1u);
    EXPECT_EQ(count_matmul(model), 0u);
}

TEST(ConvertGgmlMoeGatherToGatherMatmul, FiresOnSymmetricI4NoZeroPoint) {
    auto model = build_carrier({ov::element::i4, /*with_zp=*/false});

    ov::pass::Manager manager;
    manager.register_pass<ConvertGgmlMoeGatherToGatherMatmul>();
    manager.run_passes(model);

    EXPECT_EQ(count_gather_matmul(model), 1u);
    EXPECT_EQ(count_matmul(model), 0u);
}

// ---- End-to-end: the produced GatherMatmul folds to GatherMatmulCompressed, keeping ----
// ---- the low-bit weights compressed (no f32 materialization).                        ----

TEST(ConvertGgmlMoeGatherToGatherMatmul, FoldsToCompressedAndKeepsWeightsCompressed) {
    auto model = build_carrier({ov::element::u4, /*with_zp=*/true});

    ov::pass::Manager manager;
    manager.register_pass<ConvertGgmlMoeGatherToGatherMatmul>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertGatherMatmulToGatherMatmulCompressed>(
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f16},
        std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4},
        nullptr,
        false);
    manager.run_passes(model);

    EXPECT_EQ(count_gather_matmul_compressed(model), 1u);
    EXPECT_EQ(count_gather_matmul(model), 0u);  // folded away

    // The u4 weight Constant survives (weights stay compressed; not decompressed to f16/f32).
    size_t u4_constants = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(node); c && c->get_element_type() == ov::element::u4) {
            ++u4_constants;
        }
    }
    EXPECT_EQ(u4_constants, 1u);
}

// ---- Negative: the pass must NOT fire on graphs that are not the compressed MoE carrier ----

TEST(ConvertGgmlMoeGatherToGatherMatmul, DoesNotFireOnNonCompressedF16Weights) {
    // An f16 expert weight (plain Gather of an f16 Constant, no CompressedWeightsBlock) must
    // be left untouched — this is the case that previously crashed a broad matcher.
    const int64_t E = 8, m = 16, k = 64, n_used = 8;
    auto w = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{(size_t)E, (size_t)(m * k)},
                                          std::vector<float>((size_t)(E * m * k), 1.0f));
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, n_used});
    auto acts = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, k});

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto sel = std::make_shared<ov::op::v8::Gather>(w, ids, gather_axis);
    // Mirror the frontend: gathered weights are converted to f32 before the MatMul.
    auto sel_f32 = std::make_shared<ov::op::v0::Convert>(sel, ov::element::f32);
    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto acts_shape = std::make_shared<ov::op::v3::ShapeOf>(acts, ov::element::i64);
    auto split_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gather_dim(ids_shape, 0), gather_dim(ids_shape, 1), make_i64({m}), make_i64({-1})}, 0);
    auto selr = std::make_shared<ov::op::v1::Reshape>(sel_f32, split_dims, false);
    auto bcast_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gather_dim(acts_shape, 0), gather_dim(ids_shape, 1), gather_dim(acts_shape, 2)}, 0);
    auto acts_b = std::make_shared<ov::op::v3::Broadcast>(acts, bcast_dims, ov::op::BroadcastType::BIDIRECTIONAL);
    auto acts_e = std::make_shared<ov::op::v0::Unsqueeze>(acts_b, make_i64({2}));
    auto matmul = std::make_shared<ov::op::v0::MatMul>(acts_e, selr, false, true);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{ids, acts});

    ov::pass::Manager manager;
    manager.register_pass<ConvertGgmlMoeGatherToGatherMatmul>();
    manager.run_passes(model);

    EXPECT_EQ(count_gather_matmul(model), 0u);
    EXPECT_EQ(count_matmul(model), 1u);
}

TEST(ConvertGgmlMoeGatherToGatherMatmul, DoesNotFireWithoutTransposeB) {
    // transpose_b == false is not the expert-matmul carrier; must be left untouched.
    const int64_t E = 8, m = 16, k = 64, group = 32, n_used = 8;
    const int64_t mkg = (m * k) / group;
    auto w = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{(size_t)E, (size_t)mkg, (size_t)group},
                                          std::vector<int64_t>((size_t)(E * mkg * group), 1));
    auto cvt = std::make_shared<ov::op::v0::Convert>(w, ov::element::f16);
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{(size_t)E, (size_t)mkg, 1},
                                              std::vector<float>((size_t)(E * mkg), 0.1f));
    auto mul = std::make_shared<ov::op::v1::Multiply>(cvt, scale);
    auto block = std::make_shared<ov::op::v1::Reshape>(mul, make_i64({E, m * k}), false);
    auto block_f32 = std::make_shared<ov::op::v0::Convert>(block, ov::element::f32);
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, n_used});
    auto acts = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, m});
    auto sel = std::make_shared<ov::op::v8::Gather>(block_f32, ids,
                                                    ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto split_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gather_dim(ids_shape, 0), gather_dim(ids_shape, 1), make_i64({m}), make_i64({-1})}, 0);
    auto selr = std::make_shared<ov::op::v1::Reshape>(sel, split_dims, false);
    auto acts_shape = std::make_shared<ov::op::v3::ShapeOf>(acts, ov::element::i64);
    auto bcast_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gather_dim(acts_shape, 0), gather_dim(ids_shape, 1), gather_dim(acts_shape, 2)}, 0);
    auto acts_b = std::make_shared<ov::op::v3::Broadcast>(acts, bcast_dims, ov::op::BroadcastType::BIDIRECTIONAL);
    auto acts_e = std::make_shared<ov::op::v0::Unsqueeze>(acts_b, make_i64({2}));
    // transpose_b = false
    auto matmul = std::make_shared<ov::op::v0::MatMul>(acts_e, selr, false, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{ids, acts});

    ov::pass::Manager manager;
    manager.register_pass<ConvertGgmlMoeGatherToGatherMatmul>();
    manager.run_passes(model);

    EXPECT_EQ(count_gather_matmul(model), 0u);
    EXPECT_EQ(count_matmul(model), 1u);
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
