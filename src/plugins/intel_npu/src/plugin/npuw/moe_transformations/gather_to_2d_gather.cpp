// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_to_2d_gather.hpp"

#include <limits>
#include <numeric>
#include <optional>

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

    // Only i32/i64 indices are supported; i64 gets canonicalized to i32 in transform_gather_to_2d.
    auto indices_et = indices_input.get_element_type();
    if (indices_et != ov::element::i32 && indices_et != ov::element::i64) {
        return std::nullopt;
    }

    // Valid gather - return info
    return GatherInfo{gather, N, M, K, indices_shape[0].get_length()};
}

// Transform a single 3D Gather to 2D Gather sequence
void transform_gather_to_2d(const GatherInfo& info) {
    auto gather = info.gather_node;
    auto data_input = gather->input_value(0);
    auto indices_input = gather->input_value(1);

    std::string gather_name = gather->get_friendly_name();

    // NPU lacks efficient native i64 execution, so canonicalize indices to i32 up front.
    ov::Output<ov::Node> indices_i32 = indices_input;
    std::shared_ptr<ov::Node> indices_convert;
    if (indices_input.get_element_type() != ov::element::i32) {
        auto convert = std::make_shared<ov::op::v0::Convert>(indices_input, ov::element::i32);
        convert->set_friendly_name(gather_name + "/indices_to_i32");
        indices_convert = convert;
        indices_i32 = convert->output(0);
    }

    // Step 1: Reshape indices [I] -> [I, 1]
    std::vector<int64_t> indices_reshape_data = {static_cast<int64_t>(info.I), 1};
    auto indices_reshape_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, indices_reshape_data.data());
    auto reshaped_indices = std::make_shared<ov::op::v1::Reshape>(indices_i32, indices_reshape_shape, false);
    reshaped_indices->set_friendly_name(gather_name + "/indices_reshaped");

    // Step 2: Multiply by M to get expert starting positions [I, 1]
    auto m_const = ov::op::v0::Constant::create(ov::element::i32,
                                                ov::Shape{1, 1},
                                                std::vector<int32_t>{static_cast<int32_t>(info.M)});
    auto experts_start = std::make_shared<ov::op::v1::Multiply>(reshaped_indices, m_const);
    experts_start->set_friendly_name(gather_name + "/experts_start");

    // Step 3: Create range [0, 1, 2, ..., M-1] with shape [1, M]
    std::vector<int32_t> range_values(info.M);
    std::iota(range_values.begin(), range_values.end(), 0);
    auto range_m =
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, static_cast<size_t>(info.M)}, range_values);

    // Step 4: Add experts_start [I, 1] + range_m [1, M] -> [I, M] via numpy-style implicit broadcast
    auto new_indices = std::make_shared<ov::op::v1::Add>(experts_start, range_m);
    new_indices->set_friendly_name(gather_name + "/new_indices");

    // Step 5: Flatten indices [I, M] -> [I*M]
    std::vector<int64_t> flat_indices_shape_data = {static_cast<int64_t>(info.I * info.M)};
    auto flat_indices_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, flat_indices_shape_data.data());
    auto flat_indices = std::make_shared<ov::op::v1::Reshape>(new_indices, flat_indices_shape, false);
    flat_indices->set_friendly_name(gather_name + "/flat_indices");

    // Step 6: Flatten weights [N, M, K] -> [N*M, K]
    std::vector<int64_t> flat_weights_shape_data = {static_cast<int64_t>(info.N * info.M),
                                                    static_cast<int64_t>(info.K)};
    auto flat_weights_shape =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, flat_weights_shape_data.data());
    auto flat_weights = std::make_shared<ov::op::v1::Reshape>(data_input, flat_weights_shape, false);
    flat_weights->set_friendly_name(gather_name + "/flat_weights");

    // Step 7: Perform 2D Gather [I*M, K]
    std::vector<int64_t> gather_axis_data = {0};
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, gather_axis_data.data());
    auto gathered_flat = std::make_shared<ov::op::v8::Gather>(flat_weights, flat_indices, gather_axis);
    gathered_flat->set_friendly_name(gather_name + "/gathered_flat");

    // Step 8: Reshape to final output [I, M, K]
    std::vector<int64_t> output_shape_data = {static_cast<int64_t>(info.I),
                                              static_cast<int64_t>(info.M),
                                              static_cast<int64_t>(info.K)};
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, output_shape_data.data());
    auto final_output = std::make_shared<ov::op::v1::Reshape>(gathered_flat, output_shape, false);
    final_output->set_friendly_name(gather_name + "/output");

    // Replace the original Gather with the final Reshape
    ov::replace_node(gather, final_output);
    ov::NodeVector new_nodes{reshaped_indices,
                             experts_start,
                             range_m,
                             new_indices,
                             flat_indices,
                             flat_weights,
                             gathered_flat,
                             final_output};
    if (indices_convert) {
        new_nodes.push_back(indices_convert);
    }
    ov::copy_runtime_info(gather, new_nodes);
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

    // Transform each valid Gather
    for (const auto& info : gathers_to_transform) {
        transform_gather_to_2d(info);
    }

    if (!gathers_to_transform.empty()) {
        LOG_INFO("GatherTo2DGather: Transformed " << gathers_to_transform.size() << " Gather nodes");
    }

    return !gathers_to_transform.empty();
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
