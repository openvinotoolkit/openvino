// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_matmul_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"

namespace v0 = ov::op::v0;

namespace ov::pass {

namespace {

// Checks that removing a Broadcast placed on a MatMul input keeps the MatMul result
// unchanged. The matrix (last two) dimensions must be preserved by the Broadcast, and
// every batch dimension the Broadcast expands must be carried by the other operand.
bool can_remove_broadcast(const ov::PartialShape& data_shape,
                          const ov::PartialShape& broadcast_shape,
                          const ov::PartialShape& other_shape) {
    const int64_t data_rank = data_shape.rank().get_length();
    const int64_t broadcast_rank = broadcast_shape.rank().get_length();
    const int64_t other_rank = other_shape.rank().get_length();

    // MatMul matrix multiplication requires at least the two trailing matrix dimensions.
    if (data_rank < 2 || broadcast_rank < 2 || other_rank < 2) {
        return false;
    }

    // The Broadcast must leave the matrix (last two) dimensions of its data input intact,
    // otherwise removing it changes the contraction and the result.
    if (!ov::symbol::util::dims_are_equal(data_shape[data_rank - 1], broadcast_shape[broadcast_rank - 1]) ||
        !ov::symbol::util::dims_are_equal(data_shape[data_rank - 2], broadcast_shape[broadcast_rank - 2])) {
        return false;
    }

    // For every batch dimension of the Broadcast output (aligned from the right) either the
    // data input already carries it (Broadcast did nothing) or the other MatMul operand
    // carries it (MatMul reproduces it via implicit broadcasting). The other operand is
    // accepted when it is provably equal, or when both dimensions are dynamic (assumed
    // runtime-compatible). A fixed Broadcast extent must be matched provably, otherwise
    // detaching could hide a runtime batch mismatch the Broadcast would have rejected.
    const int64_t broadcast_batch = broadcast_rank - 2;
    for (int64_t offset = 0; offset < broadcast_batch; ++offset) {
        const ov::Dimension& broadcast_dim = broadcast_shape[broadcast_batch - 1 - offset];

        const int64_t data_idx = (data_rank - 2) - 1 - offset;
        if (data_idx >= 0 && ov::symbol::util::dims_are_equal(data_shape[data_idx], broadcast_dim)) {
            continue;
        }

        const int64_t other_idx = (other_rank - 2) - 1 - offset;
        if (other_idx < 0) {
            return false;
        }
        const ov::Dimension& other_dim = other_shape[other_idx];
        if (ov::symbol::util::dims_are_equal(other_dim, broadcast_dim)) {
            continue;
        }
        if (broadcast_dim.is_dynamic() && other_dim.is_dynamic()) {
            continue;
        }
        return false;
    }

    return true;
}

}  // namespace

BroadcastMatMulFusion::BroadcastMatMulFusion() {
    MATCHER_SCOPE(BroadcastMatMulFusion);

    // Match Constant -> Broadcast -> MatMul with the Broadcast on either MatMul input.
    auto data = pattern::wrap_type<v0::Constant>(pattern::has_static_rank());
    auto broadcast =
        pattern::wrap_type<ov::op::util::BroadcastBase>({data, pattern::any_input()},
                                                        pattern::consumers_count(1) && pattern::has_static_rank());
    auto other = pattern::any_input(pattern::has_static_rank());
    auto matmul_lhs = pattern::wrap_type<v0::MatMul>({broadcast, other});
    auto matmul_rhs = pattern::wrap_type<v0::MatMul>({other, broadcast});
    auto matmul = std::make_shared<pattern::op::Or>(OutputVector{matmul_lhs, matmul_rhs});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto broadcast_value = pattern_map.at(broadcast);
        const auto broadcast_op = ov::as_type_ptr<ov::op::util::BroadcastBase>(broadcast_value.get_node_shared_ptr());

        // Only NumPy-style broadcasting matches MatMul's implicit batch broadcasting.
        // PDPD / EXPLICIT modes align dimensions differently and must not be detached.
        const ov::op::BroadcastType mode = broadcast_op->get_broadcast_spec().m_type;
        if (mode != ov::op::BroadcastType::NUMPY && mode != ov::op::BroadcastType::BIDIRECTIONAL) {
            return false;
        }

        const auto matmul_node = m.get_match_root();

        const size_t broadcast_port = matmul_node->input_value(0) == broadcast_value ? 0 : 1;
        const auto data_value = pattern_map.at(data);
        const auto other_value = matmul_node->input_value(1 - broadcast_port);

        if (!can_remove_broadcast(data_value.get_partial_shape(),
                                  broadcast_value.get_partial_shape(),
                                  other_value.get_partial_shape())) {
            return false;
        }

        copy_runtime_info(broadcast_value.get_node_shared_ptr(), matmul_node);
        matmul_node->input(broadcast_port).replace_source_output(data_value);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(matmul, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
