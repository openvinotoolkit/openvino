// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_to_2d_gather.hpp"

#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <unordered_map>

#include "../logging.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace {

// ============================================================================
// Helper structures for organizing transformation data
// ============================================================================

struct GatherInfo {
    std::shared_ptr<ov::op::v8::Gather> gather_node;
    int64_t N;  // num_experts (data dim 0)
    int64_t M;  // feature_dim (data dim 1)
    int64_t K;  // hidden_dim (data dim 2)
    int64_t I;  // num_selected (indices size)
};

// ============================================================================
// Helper functions
// ============================================================================

// Check if a Gather node is valid for 3D->2D transformation
std::optional<GatherInfo> validate_gather_for_transform(const std::shared_ptr<ov::op::v8::Gather>& gather) {
    if (!gather) {
        return std::nullopt;
    }

    // Get gather inputs
    auto data_input = gather->input_value(0);
    auto indices_input = gather->input_value(1);
    auto axis_input = gather->input_value(2);

    // Check if axis is 0 (gathering on first dimension)
    auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(axis_input.get_node_shared_ptr());
    if (!axis_const) {
        return std::nullopt;
    }
    auto axis_value = axis_const->cast_vector<int64_t>()[0];
    if (axis_value != 0) {
        return std::nullopt;
    }

    // Check data shape: should be 3D [N, M, K] with static dimensions
    auto data_shape = data_input.get_partial_shape();
    if (!data_shape.rank().is_static() || data_shape.rank().get_length() != 3) {
        return std::nullopt;
    }
    if (!data_shape[0].is_static() || !data_shape[1].is_static() || !data_shape[2].is_static()) {
        return std::nullopt;
    }

    int64_t N = data_shape[0].get_length();
    int64_t M = data_shape[1].get_length();
    int64_t K = data_shape[2].get_length();

    // Only transform if both M and K are not 1 (otherwise transformation is not beneficial)
    if (M == 1 || K == 1) {
        return std::nullopt;
    }

    // Flattened indices are computed as (index * M + [0, M)) in i32; skip the rewrite if N*M
    // can't be represented in i32 to avoid silently wrong results from overflow.
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    if (M != 0 && N > kInt32Max / M) {
        return std::nullopt;
    }

    // Check indices shape: should be 1D [I] with static dimension
    auto indices_shape = indices_input.get_partial_shape();
    if (!indices_shape.rank().is_static() || indices_shape.rank().get_length() != 1) {
        return std::nullopt;
    }
    if (!indices_shape[0].is_static()) {
        return std::nullopt;
    }

    // Only i32/i64 indices are supported; i64 gets canonicalized to i32 in build_shared_new_indices.
    auto indices_et = indices_input.get_element_type();
    if (indices_et != ov::element::i32 && indices_et != ov::element::i64) {
        return std::nullopt;
    }

    // Valid gather - return info
    return GatherInfo{gather, N, M, K, indices_shape[0].get_length()};
}

// Sibling Gathers that read the same indices tensor share an identical indices-transform
// prefix, so it should be built once per group instead of once per Gather. This is the common
// case in MoE experts, where the same selected-expert indices feed several sibling Gathers
// (e.g. gate_proj/up_proj/down_proj).

// Groups Gathers by indices source: same source Output -> same group. Preserves topological
// order both across groups (order first encountered) and within a group.
//
// Key: the indices Output itself (producer node + output index) identifies the source
// directly, since it already supports ordering comparison -- no custom key/hash type needed.
std::vector<std::vector<GatherInfo>> group_by_indices_source(const std::vector<GatherInfo>& gathers) {
    std::vector<std::vector<GatherInfo>> groups;
    std::map<ov::Output<ov::Node>, size_t> indices_to_group;

    for (const auto& info : gathers) {
        auto indices_input = info.gather_node->input_value(1);
        auto it = indices_to_group.find(indices_input);
        if (it == indices_to_group.end()) {
            indices_to_group[indices_input] = groups.size();
            groups.push_back({info});
        } else {
            groups[it->second].push_back(info);
        }
    }
    return groups;
}

// Build the indices-transform prefix (steps 0-4) once for a group of Gathers that all read the
// same indices tensor, then attribute every node created here to every Gather in the group via
// copy_runtime_info. Members are deduplicated by M:
//  - Same M (including a singleton group): every member reuses one scalar Multiply/Add, byte-
//    for-byte identical to the original per-Gather construction.
//  - Different M: one combined Multiply/Add is computed over per-branch M/range values
//    concatenated at pass-build time (plain std::vector, not a runtime Concat), then routed to
//    each branch via VariadicSplit.
//
// Returns: one Output per member of `group` (same order) -- the [I, M_i] tensor that member
// should flatten and gather with.
std::vector<ov::Output<ov::Node>> build_shared_new_indices(const std::vector<GatherInfo>& group) {
    OPENVINO_ASSERT(!group.empty());
    auto indices_input = group.front().gather_node->input_value(1);
    int64_t I = group.front().I;

    // Name new nodes after the shared indices source (siblings have no single owning Gather);
    // a singleton group falls back to the Gather's own name, matching the pre-sharing naming.
    std::string base_name = group.size() == 1 ? group.front().gather_node->get_friendly_name()
                                              : indices_input.get_node()->get_friendly_name() + "/shared_experts";

    ov::NodeVector created;

    // Step 0: canonicalize indices to i32 once (NPU lacks efficient native i64 execution).
    ov::Output<ov::Node> indices_i32 = indices_input;
    if (indices_input.get_element_type() != ov::element::i32) {
        auto convert = std::make_shared<ov::op::v0::Convert>(indices_input, ov::element::i32);
        convert->set_friendly_name(base_name + "/indices_to_i32");
        created.push_back(convert);
        indices_i32 = convert->output(0);
    }

    // Step 1: Reshape indices [I] -> [I, 1] -- shared by every member since I is identical
    // across the whole group (they all read the very same indices tensor).
    auto indices_reshape_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{I, 1});
    auto reshaped_indices = std::make_shared<ov::op::v1::Reshape>(indices_i32, indices_reshape_shape, false);
    reshaped_indices->set_friendly_name(base_name + "/indices_reshaped");
    created.push_back(reshaped_indices);

    // Deduplicate by M, preserving first-seen order.
    std::vector<int64_t> distinct_m;
    std::unordered_map<int64_t, size_t> m_to_slot;
    std::vector<size_t> branch_slot(group.size());
    for (size_t i = 0; i < group.size(); ++i) {
        int64_t M = group[i].M;
        auto it = m_to_slot.find(M);
        if (it == m_to_slot.end()) {
            branch_slot[i] = distinct_m.size();
            m_to_slot[M] = branch_slot[i];
            distinct_m.push_back(M);
        } else {
            branch_slot[i] = it->second;
        }
    }

    std::vector<ov::Output<ov::Node>> new_indices_per_branch(group.size());

    if (distinct_m.size() == 1) {
        // Same M for every member (also the only case a singleton group hits):
        //   1. experts_start = indices[I,1] * M            (Multiply, scalar M)
        //   2. new_indices   = experts_start + range[0..M) (Add)
        // new_indices is [I, M] and is reused as-is by every branch -- no Split needed.
        int64_t M = distinct_m.front();
        auto m_const = ov::op::v0::Constant::create(ov::element::i32,
                                                    ov::Shape{1, 1},
                                                    std::vector<int32_t>{static_cast<int32_t>(M)});
        auto experts_start = std::make_shared<ov::op::v1::Multiply>(reshaped_indices, m_const);
        experts_start->set_friendly_name(base_name + "/experts_start");
        created.push_back(experts_start);

        std::vector<int32_t> range_values(static_cast<size_t>(M));
        std::iota(range_values.begin(), range_values.end(), 0);
        auto range_m =
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, static_cast<size_t>(M)}, range_values);

        auto new_indices = std::make_shared<ov::op::v1::Add>(experts_start, range_m);
        new_indices->set_friendly_name(base_name + "/new_indices");
        created.push_back(new_indices);

        for (size_t i = 0; i < group.size(); ++i) {
            new_indices_per_branch[i] = new_indices->output(0);
        }
    } else {
        // Mixed M values: compute every branch's formula side by side in one wide tensor, then
        // split it back into one chunk per distinct M.
        //   m_vec[j]     = M of the block column j belongs to  (per-column Multiply factor)
        //   range_vec[j] = local offset of column j in its block (0..M-1)
        //
        //   1. experts_start        = indices[I,1] * m_vec              (Multiply)
        //   2. new_indices_combined = experts_start + range_vec         (Add)   -> [I, sum(distinct_m)]
        //   3. per-branch chunks    = VariadicSplit(new_indices_combined, by distinct_m)
        std::vector<int32_t> m_vec_data;
        std::vector<int32_t> range_vec_data;
        for (int64_t M : distinct_m) {
            m_vec_data.insert(m_vec_data.end(), static_cast<size_t>(M), static_cast<int32_t>(M));
            for (int64_t j = 0; j < M; ++j) {
                range_vec_data.push_back(static_cast<int32_t>(j));
            }
        }
        size_t combined_len = m_vec_data.size();

        auto m_vec_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, combined_len}, m_vec_data);
        auto experts_start = std::make_shared<ov::op::v1::Multiply>(reshaped_indices, m_vec_const);
        experts_start->set_friendly_name(base_name + "/experts_start");
        created.push_back(experts_start);

        auto range_vec_const =
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, combined_len}, range_vec_data);
        auto new_indices_combined = std::make_shared<ov::op::v1::Add>(experts_start, range_vec_const);
        new_indices_combined->set_friendly_name(base_name + "/new_indices");
        created.push_back(new_indices_combined);

        std::vector<int64_t> split_lengths(distinct_m.begin(), distinct_m.end());
        auto split_lengths_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths);
        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(new_indices_combined, split_axis, split_lengths_const);
        split->set_friendly_name(base_name + "/split_by_m");
        created.push_back(split);

        for (size_t i = 0; i < group.size(); ++i) {
            new_indices_per_branch[i] = split->output(branch_slot[i]);
        }
    }

    ov::NodeVector group_gathers;
    group_gathers.reserve(group.size());
    for (const auto& info : group) {
        group_gathers.push_back(info.gather_node);
    }
    ov::copy_runtime_info(group_gathers, created);

    return new_indices_per_branch;
}

// Build one branch's Gather itself (steps 5-8: flatten indices/weights, 2D Gather, reshape
// back to [I, M, K]), given the new_indices this branch was assigned by the shared prefix.
void build_gather_suffix(const GatherInfo& info, const ov::Output<ov::Node>& new_indices) {
    auto gather = info.gather_node;
    auto data_input = gather->input_value(0);
    std::string gather_name = gather->get_friendly_name();

    // Step 5: Flatten indices [I, M] -> [I*M]
    auto flat_indices_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{info.I * info.M});
    auto flat_indices = std::make_shared<ov::op::v1::Reshape>(new_indices, flat_indices_shape, false);
    flat_indices->set_friendly_name(gather_name + "/flat_indices");

    // Step 6: Flatten weights [N, M, K] -> [N*M, K]
    auto flat_weights_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{info.N * info.M, info.K});
    auto flat_weights = std::make_shared<ov::op::v1::Reshape>(data_input, flat_weights_shape, false);
    flat_weights->set_friendly_name(gather_name + "/flat_weights");

    // Step 7: Perform 2D Gather [I*M, K]
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto gathered_flat = std::make_shared<ov::op::v8::Gather>(flat_weights, flat_indices, gather_axis);
    gathered_flat->set_friendly_name(gather_name + "/gathered_flat");

    // Step 8: Reshape to final output [I, M, K]
    auto output_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{info.I, info.M, info.K});
    auto final_output = std::make_shared<ov::op::v1::Reshape>(gathered_flat, output_shape, false);
    final_output->set_friendly_name(gather_name + "/output");

    // Replace the original Gather with the final Reshape
    ov::replace_node(gather, final_output);
    ov::copy_runtime_info(gather, {flat_indices, flat_weights, gathered_flat, final_output});
}

// Transform a whole group of Gathers that share the same indices source: build the shared
// prefix once, then let every member build its own suffix off the slice it was assigned.
void transform_gather_group(const std::vector<GatherInfo>& group) {
    auto new_indices_per_branch = build_shared_new_indices(group);
    for (size_t i = 0; i < group.size(); ++i) {
        build_gather_suffix(group[i], new_indices_per_branch[i]);
    }
}

}  // anonymous namespace

// ============================================================================
// Main transformation entry point
// ============================================================================

bool GatherTo2DGather::run_on_model(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("GatherTo2DGather: Starting transformation");

    std::vector<GatherInfo> gathers_to_transform;

    // Collect and validate Gather nodes
    for (const auto& node : model->get_ordered_ops()) {
        auto gather = std::dynamic_pointer_cast<ov::op::v8::Gather>(node);
        auto gather_info = validate_gather_for_transform(gather);

        if (gather_info.has_value()) {
            gathers_to_transform.push_back(gather_info.value());
        }
    }

    // Transform each group of Gathers sharing the same indices source
    auto groups = group_by_indices_source(gathers_to_transform);
    for (const auto& group : groups) {
        transform_gather_group(group);
    }

    if (!gathers_to_transform.empty()) {
        LOG_INFO("GatherTo2DGather: Transformed " << gathers_to_transform.size() << " Gather node(s) across "
                                                  << groups.size() << " shared indices group(s)");
    }

    return !gathers_to_transform.empty();
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
