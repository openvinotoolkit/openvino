// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_gather_matmul.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/search_sorted.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;
namespace v15 = ov::op::v15;
namespace v17 = ov::op::v17;

using ov::op::internal::GatherMatmul;

static bool is_empty_pshape(const ov::PartialShape& ps) {
    return ps.rank().is_static() && ps.rank().get_length() == 1 && ps[0].is_static() &&
           ps[0].get_length() == 0;
}

// Build the identity-per-group `indices` for Case 2:
//     indices = Broadcast(Unsqueeze(Range(0, G), 0), [M, G])
// The G and M dimensions are taken dynamically from ShapeOf(mat_a).
ov::Output<ov::Node> build_case2_indices(const ov::Output<ov::Node>& mat_a, ov::NodeVector& new_nodes) {
    auto i32 = ov::element::i32;

    auto shape_a = std::make_shared<v3::ShapeOf>(mat_a, i32);  // [3]: [G, M, K]
    auto axis0 = v0::Constant::create(i32, ov::Shape{}, {0});
    auto idx_g = v0::Constant::create(i32, ov::Shape{1}, {0});
    auto idx_m = v0::Constant::create(i32, ov::Shape{1}, {1});
    auto g_dim = std::make_shared<v8::Gather>(shape_a, idx_g, axis0);  // [1]
    auto m_dim = std::make_shared<v8::Gather>(shape_a, idx_m, axis0);  // [1]

    // Build target shape [M, G] for broadcasting.
    auto target_shape = std::make_shared<v0::Concat>(ov::OutputVector{m_dim, g_dim}, 0);

    // Range(0, G, 1) -> [G].  Range needs scalar inputs of matching type.
    auto g_scalar_idx = v0::Constant::create(i32, ov::Shape{}, {0});
    auto g_scalar = std::make_shared<v8::Gather>(shape_a, g_scalar_idx, axis0);  // scalar
    auto zero = v0::Constant::create(i32, ov::Shape{}, {0});
    auto one = v0::Constant::create(i32, ov::Shape{}, {1});
    auto range = std::make_shared<v4::Range>(zero, g_scalar, one, i32);  // [G]

    // [G] -> [1, G]
    auto unsqueeze_axis = v0::Constant::create(i32, ov::Shape{1}, {0});
    auto range_2d = std::make_shared<v0::Unsqueeze>(range, unsqueeze_axis);

    // Broadcast to [M, G].  Each row is identity 0..G-1, so indices[m, g] = g.
    auto indices = std::make_shared<v3::Broadcast>(range_2d, target_shape);

    new_nodes.insert(new_nodes.end(),
                     {shape_a, axis0, idx_g, idx_m, g_dim, m_dim, target_shape,
                      g_scalar_idx, g_scalar, zero, one, range, unsqueeze_axis, range_2d, indices});
    return indices;
}

// Build per-token expert indices from `offsets` for Case 1:
//     positions  = Range(0, T)                                     [T]
//     idx_1d     = SearchSorted(offsets, positions, right=true)    [T]
//     indices    = Unsqueeze(idx_1d, -1)                           [T, 1]
//
// `offsets` may be i32 or i64; we Convert to i32 and use i32 positions to keep
// SearchSorted's type contract happy and to match GatherMatmul's preferred index type.
ov::Output<ov::Node> build_case1_indices(const ov::Output<ov::Node>& mat_a,
                                         const ov::Output<ov::Node>& offsets,
                                         ov::NodeVector& new_nodes) {
    auto i32 = ov::element::i32;

    auto offsets_i32 = std::make_shared<v0::Convert>(offsets, i32);

    auto shape_a = std::make_shared<v3::ShapeOf>(mat_a, i32);  // [2]: [T, K]
    auto axis0 = v0::Constant::create(i32, ov::Shape{}, {0});
    auto t_idx = v0::Constant::create(i32, ov::Shape{}, {0});
    auto t_scalar = std::make_shared<v8::Gather>(shape_a, t_idx, axis0);  // scalar T
    auto zero = v0::Constant::create(i32, ov::Shape{}, {0});
    auto one = v0::Constant::create(i32, ov::Shape{}, {1});
    auto positions = std::make_shared<v4::Range>(zero, t_scalar, one, i32);  // [T]

    auto idx_1d = std::make_shared<v15::SearchSorted>(offsets_i32, positions, /*right_mode=*/true, i32);

    auto unsqueeze_axis = v0::Constant::create(i32, ov::Shape{1}, {-1});
    auto indices = std::make_shared<v0::Unsqueeze>(idx_1d, unsqueeze_axis);

    new_nodes.insert(new_nodes.end(),
                     {offsets_i32, shape_a, axis0, t_idx, t_scalar, zero, one, positions, idx_1d, unsqueeze_axis,
                      indices});
    return indices;
}

}  // namespace

ConvertGroupedMatMulToGatherMatmul::ConvertGroupedMatMulToGatherMatmul() {
    MATCHER_SCOPE(ConvertGroupedMatMulToGatherMatmul);

    auto gmm_m = ov::pass::pattern::wrap_type<v17::GroupedMatMul>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gmm = ov::as_type_ptr<v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }

        const auto& a_pshape = gmm->get_input_partial_shape(0);
        const auto& b_pshape = gmm->get_input_partial_shape(1);
        if (a_pshape.rank().is_dynamic() || b_pshape.rank().is_dynamic()) {
            return false;
        }
        const size_t a_rank = a_pshape.size();
        const size_t b_rank = b_pshape.size();

        const auto mat_a = gmm->input_value(0);
        const auto mat_b = gmm->input_value(1);

        // GatherMatmul requires the weights tensor (input B) to be a Constant
        // (or reachable through a constant-foldable chain). If it isn't, we cannot
        // lower to the internal op — leave the public op in place so the reference
        // GroupedMatMul::evaluate is used instead.
        if (!ov::op::util::is_on_path<v0::Constant>(mat_b)) {
            return false;
        }

        const auto& bias_pshape = gmm->get_input_partial_shape(3);
        const bool bias_is_empty = is_empty_pshape(bias_pshape);
        const auto bias_val = gmm->input_value(3);

        ov::NodeVector new_nodes;
        std::shared_ptr<ov::Node> replacement;

        if (a_rank == 3 && b_rank == 3) {
            // ---- Case 2: 3D x 3D, no offsets ----
            // A:[G,M,K] B:[G,N,K] -> GatherMatmul(A, B, indices=[M,G]) -> [G,M,N]
            auto indices = build_case2_indices(mat_a, new_nodes);
            auto gm = std::make_shared<GatherMatmul>(mat_a, mat_b, indices);
            new_nodes.push_back(gm);
            replacement = gm;

            if (!bias_is_empty) {
                // bias [G, N] -> Unsqueeze(axis=1) -> [G, 1, N] -> Add to [G, M, N]
                auto unsq_axis = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
                auto bias_3d = std::make_shared<v0::Unsqueeze>(bias_val, unsq_axis);
                auto with_bias = std::make_shared<v1::Add>(replacement, bias_3d);
                new_nodes.push_back(unsq_axis);
                new_nodes.push_back(bias_3d);
                new_nodes.push_back(with_bias);
                replacement = with_bias;
            }
        } else if (a_rank == 2 && b_rank == 3) {
            // ---- Case 1: 2D x 3D with offsets ----
            // A:[T,K] B:[G,N,K] offs:[G] -> Squeeze(GatherMatmul(Unsqueeze(A,0), B, idx[T,1]), 0) -> [T,N]
            const auto offsets = gmm->input_value(2);

            auto a_unsq_axis = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
            auto a_3d = std::make_shared<v0::Unsqueeze>(mat_a, a_unsq_axis);
            new_nodes.push_back(a_unsq_axis);
            new_nodes.push_back(a_3d);

            auto indices = build_case1_indices(mat_a, offsets, new_nodes);

            auto gm = std::make_shared<GatherMatmul>(a_3d, mat_b, indices);   // [1, T, N]
            auto squeeze_axis = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
            auto out = std::make_shared<v0::Squeeze>(gm, squeeze_axis);     // [T, N]

            new_nodes.push_back(gm);
            new_nodes.push_back(squeeze_axis);
            new_nodes.push_back(out);
            replacement = out;

            if (!bias_is_empty) {
                // indices is [T, 1]; squeeze axis=1 to get idx_1d [T]
                // then Gather(bias[G,N], idx_1d[T], axis=0) -> [T, N] -> Add to output [T, N]
                auto sq_axis = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
                auto idx_1d = std::make_shared<v0::Squeeze>(indices, sq_axis);
                auto axis0 = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
                auto bias_per_token = std::make_shared<v8::Gather>(bias_val, idx_1d, axis0);
                auto with_bias = std::make_shared<v1::Add>(replacement, bias_per_token);
                new_nodes.push_back(sq_axis);
                new_nodes.push_back(idx_1d);
                new_nodes.push_back(axis0);
                new_nodes.push_back(bias_per_token);
                new_nodes.push_back(with_bias);
                replacement = with_bias;
            }
        } else {
            // Case 3 (2D x 2D weight gradient) and any unexpected shape combinations
            // are not handled by GatherMatmul — leave the graph untouched so the
            // default decomposition / reference path can take over.
            return false;
        }

        replacement->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(gmm, new_nodes);
        ov::replace_node(gmm, replacement);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_m, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
