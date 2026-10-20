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
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/search_sorted.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
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

// Build the identity-per-group `indices` for 3Dx3D:
//     indices = Broadcast(Range(0, G), [M, G])
// The G and M dimensions are taken dynamically from ShapeOf(mat_a).
ov::Output<ov::Node> build_indices_for_3dx3d(const ov::Output<ov::Node>& mat_a, NodeRegistry& rg) {
    auto i32 = ov::element::i32;
    auto zero = rg.make<v0::Constant>(i32, ov::Shape{}, 0);
    auto shape_a = rg.make<v3::ShapeOf>(mat_a, i32);  // [G, M, K]

    // Single Gather produces the Broadcast target shape [M, G] directly
    auto mg_idx = rg.make<v0::Constant>(i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
    auto target_shape = rg.make<v8::Gather>(shape_a, mg_idx, zero);  // [M, G]

    // Scalar G for Range stop (index 0 along axis 0).
    auto g_scalar = rg.make<v8::Gather>(shape_a, zero, zero);
    auto one = rg.make<v0::Constant>(i32, ov::Shape{}, 1);
    auto range = rg.make<v4::Range>(zero, g_scalar, one, i32);  // [G]

    // Broadcast right-aligns [G] against the target [M, G]
    auto indices = rg.make<v3::Broadcast>(range, target_shape);
    return indices;
}

// Build per-token expert indices from `offsets` for 2Dx3D:
//     positions  = Range(0, T)                                     [T]
//     idx_1d     = SearchSorted(offsets, positions, right=true)    [T]
//     indices    = Unsqueeze(idx_1d, -1)                           [T, 1]
//
// `offsets` may be i32 or i64; we Convert to i32 only when needed and use i32
ov::Output<ov::Node> build_indices_for_2dx3d(const ov::Output<ov::Node>& mat_a,
                                             const ov::Output<ov::Node>& offsets,
                                             NodeRegistry& rg) {
    auto i32 = ov::element::i32;
    auto zero = rg.make<v0::Constant>(i32, ov::Shape{}, 0);

    // SearchSorted requires the sorted sequence and probe values to share an element
    // type; convert `offsets` to i32 only when it isn't already i32.
    ov::Output<ov::Node> offsets_i32 = offsets;
    if (offsets.get_element_type() != i32) {
        offsets_i32 = rg.make<v0::Convert>(offsets, i32);
    }

    auto shape_a = rg.make<v3::ShapeOf>(mat_a, i32);           // [T, K]
    auto t_scalar = rg.make<v8::Gather>(shape_a, zero, zero);  // scalar T
    auto one = rg.make<v0::Constant>(i32, ov::Shape{}, 1);
    auto positions = rg.make<v4::Range>(zero, t_scalar, one, i32);  // [T]
    auto idx_1d = rg.make<v15::SearchSorted>(offsets_i32, positions, /*right_mode=*/true, i32);
    auto unsqueeze_axis = rg.make<v0::Constant>(i32, ov::Shape{1}, -1);
    auto indices = rg.make<v0::Unsqueeze>(idx_1d, unsqueeze_axis);
    return indices;
}

}  // namespace

ConvertGroupedMatMulToGatherMatmul::ConvertGroupedMatMulToGatherMatmul() {
    MATCHER_SCOPE(ConvertGroupedMatMulToGatherMatmul);
    using namespace ov::pass::pattern;

    // Common pattern for second input
    auto matrix_b_3d = any_input(rank_equals(3));

    // ---- 3Dx3D: no offsets ----
    // A:[G,M,K] B:[G,N,K]
    auto matrix_a_3d = any_input(rank_equals(3));
    auto gmm_3d_3d = wrap_type<v17::GroupedMatMul>({matrix_a_3d, matrix_b_3d});

    // ---- 2Dx3D: with offsets ----
    // A:[T,K] B:[G,N,K] offsets:[G]
    auto matrix_a_2d = any_input(rank_equals(2));
    auto offsets = any_input(rank_equals(1));
    auto gmm_2d_3d = wrap_type<v17::GroupedMatMul>({matrix_a_2d, matrix_b_3d, offsets});

    auto gmm_pattern = gmm_3d_3d | gmm_2d_3d;

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gmm = ov::as_type_ptr<v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }
        const auto mat_a = gmm->input_value(0);
        const auto mat_b = gmm->input_value(1);

        // GatherMatmul requires the weights tensor (input B) to be a Constant
        // (or reachable through a constant-foldable chain).
        if (!ov::op::util::is_on_path<v0::Constant>(mat_b)) {
            return false;
        }

        NodeRegistry rg;
        std::shared_ptr<ov::Node> replacement;
        const auto& pattern_map = m.get_pattern_value_map();
        if (pattern_map.count(gmm_3d_3d)) {
            // ---- 3Dx3D: no offsets ----
            // A:[G,M,K] B:[G,N,K] -> GatherMatmul(A, B, indices=[M,G]) -> [G,M,N]
            auto indices = build_indices_for_3dx3d(mat_a, rg);
            replacement = rg.make<GatherMatmul>(mat_a, mat_b, indices);
        } else if (pattern_map.count(gmm_2d_3d)) {
            // ---- 2Dx3D: with offsets ----
            // A:[T,K] B:[G,N,K] offs:[G] -> Squeeze(GatherMatmul(Unsqueeze(A,0), B, idx[T,1]), 0) -> [T,N]
            const auto offsets = gmm->input_value(2);
            auto a_unsq_axis = rg.make<v0::Constant>(ov::element::i32, ov::Shape{1}, 0);
            auto a_3d = rg.make<v0::Unsqueeze>(mat_a, a_unsq_axis);
            auto indices = build_indices_for_2dx3d(mat_a, offsets, rg);
            auto gm = rg.make<GatherMatmul>(a_3d, mat_b, indices);  // [1, T, N]
            auto squeeze_axis = rg.make<v0::Constant>(ov::element::i32, ov::Shape{1}, 0);
            replacement = rg.make<v0::Squeeze>(gm, squeeze_axis);  // [T, N]
        } else {
            return false;
        }

        replacement->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(gmm, rg.get());
        ov::replace_node(gmm, replacement);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_pattern, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
