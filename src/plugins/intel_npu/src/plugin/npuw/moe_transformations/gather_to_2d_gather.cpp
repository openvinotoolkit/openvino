// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_to_2d_gather.hpp"

#include "../logging.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"

namespace ov {
namespace npuw {
namespace pass {

/**
 * @brief Transform 3D Gather to 2D Gather sequence
 *
 * Original: Gather(weights[N, M, K], indices[I], axis=0) -> [I, M, K]
 *
 * Transformed sequence:
 * 1. reshaped_indices[I,1] = Reshape(indices[I])
 * 2. experts_start[I,1] = Multiply(reshaped_indices, Constant(M))
 * 3. range_M = Constant[[0,1,2,...,M-1], ...] shape [I, M] (tiled)
 * 4. new_indices[I,M] = Add(experts_start, range_M)
 * 5. flat_indices[I*M] = Reshape(new_indices)
 * 6. flat_weights[N*M, K] = Reshape(weights)
 * 7. gathered[I*M, K] = Gather(flat_weights, flat_indices, axis=0)
 * 8. output[I, M, K] = Reshape(gathered)
 */
bool GatherTo2DGather::run_on_model(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("GatherTo2DGather: Starting transformation");
    std::cout << "GatherTo2DGather: Starting transformation" << std::endl;

    bool model_changed = false;
    std::vector<std::shared_ptr<ov::op::v8::Gather>> gathers_to_transform;

    // Collect Gather nodes that need transformation
    for (const auto& node : model->get_ordered_ops()) {
        auto gather = std::dynamic_pointer_cast<ov::op::v8::Gather>(node);
        if (!gather) {
            continue;
        }

        // Get gather inputs
        auto data_input = gather->input_value(0);
        auto indices_input = gather->input_value(1);
        auto axis_input = gather->input_value(2);

        // Check if axis is 0 (gathering on first dimension)
        auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(axis_input.get_node_shared_ptr());
        if (!axis_const) {
            continue;
        }
        auto axis_value = axis_const->cast_vector<int64_t>()[0];
        if (axis_value != 0) {
            continue;
        }

        // Check data shape: should be 3D [N, M, K]
        auto data_shape = data_input.get_partial_shape();
        if (!data_shape.rank().is_static() || data_shape.rank().get_length() != 3) {
            continue;
        }
        if (!data_shape[0].is_static() || !data_shape[1].is_static() || !data_shape[2].is_static()) {
            continue;
        }

        // Only transform if both M and K are not 1 (otherwise transformation is not beneficial)
        int64_t M = data_shape[1].get_length();
        int64_t K = data_shape[2].get_length();
        if (M == 1 || K == 1) {
            std::cout << "  Skipping Gather with shape [_, " << M << ", " << K
                      << "]: transformation not needed for trivial dimensions" << std::endl;
            continue;
        }

        // Check indices shape: should be 1D [I]
        auto indices_shape = indices_input.get_partial_shape();
        if (!indices_shape.rank().is_static() || indices_shape.rank().get_length() != 1) {
            continue;
        }
        if (!indices_shape[0].is_static()) {
            continue;
        }

        std::cout << "  Found Gather to transform: " << gather->get_friendly_name() << " data=" << data_shape
                  << " indices=" << indices_shape << std::endl;
        gathers_to_transform.push_back(gather);
    }

    // Transform each collected Gather
    for (auto& gather : gathers_to_transform) {
        auto data_input = gather->input_value(0);
        auto indices_input = gather->input_value(1);

        auto data_shape = data_input.get_partial_shape();
        auto indices_shape = indices_input.get_partial_shape();

        int64_t N = data_shape[0].get_length();     // num_experts
        int64_t M = data_shape[1].get_length();     // feature_dim
        int64_t K = data_shape[2].get_length();     // hidden_dim
        int64_t I = indices_shape[0].get_length();  // num selected (K value)

        std::string gather_name = gather->get_friendly_name();

        // Step 1: Reshape indices [I] -> [I, 1]
        std::vector<int64_t> indices_reshape_data = {I, 1};
        auto indices_reshape_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, indices_reshape_data);
        auto reshaped_indices = std::make_shared<ov::op::v1::Reshape>(indices_input, indices_reshape_shape, false);
        reshaped_indices->set_friendly_name(gather_name + "/indices_reshaped");

        // Step 2: Multiply by M to get expert starting positions [I, 1]
        std::vector<int64_t> m_data = {M};
        auto m_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1}, m_data);
        auto experts_start = std::make_shared<ov::op::v1::Multiply>(reshaped_indices, m_const);
        experts_start->set_friendly_name(gather_name + "/experts_start");

        // Step 3: Create range [0, 1, 2, ..., M-1] and tile to [I, M]
        std::vector<int64_t> range_values(M);
        for (int64_t i = 0; i < M; ++i) {
            range_values[i] = i;
        }
        auto range_m =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, static_cast<size_t>(M)}, range_values);
        // Mark this constant to be preserved in function body during partitioning.
        // This range constant is identical across all repeated MoE layers and should not be parameterized
        // to avoid unnecessary overhead. The partitioning pass will check this marker to keep it in-place.
        range_m->get_rt_info()["npuw_moe_gather_indices"] = true;

        // Tile range to [I, M]
        std::vector<int64_t> tile_repeats_data = {I, 1};
        auto tile_repeats = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, tile_repeats_data);
        auto range_m_tiled = std::make_shared<ov::op::v0::Tile>(range_m, tile_repeats);
        range_m_tiled->set_friendly_name(gather_name + "/range_tiled");

        // Add experts_start + range to get final indices [I, M]
        auto new_indices = std::make_shared<ov::op::v1::Add>(experts_start, range_m_tiled);
        new_indices->set_friendly_name(gather_name + "/new_indices");

        // Step 4: Flatten indices [I, M] -> [I*M]
        std::vector<int64_t> flat_indices_shape_data = {I * M};
        auto flat_indices_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, flat_indices_shape_data);
        auto flat_indices = std::make_shared<ov::op::v1::Reshape>(new_indices, flat_indices_shape, false);
        flat_indices->set_friendly_name(gather_name + "/flat_indices");

        // Step 5: Flatten weights [N, M, K] -> [N*M, K]
        std::vector<int64_t> flat_weights_shape_data = {N * M, K};
        auto flat_weights_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, flat_weights_shape_data);
        auto flat_weights = std::make_shared<ov::op::v1::Reshape>(data_input, flat_weights_shape, false);
        flat_weights->set_friendly_name(gather_name + "/flat_weights");

        std::vector<int64_t> gather_axis_data = {0};
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, gather_axis_data);
        auto gathered_flat = std::make_shared<ov::op::v8::Gather>(flat_weights, flat_indices, gather_axis);
        gathered_flat->set_friendly_name(gather_name + "/gathered_flat");

        std::vector<int64_t> output_shape_data = {I, M, K};
        auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, output_shape_data);
        auto final_output = std::make_shared<ov::op::v1::Reshape>(gathered_flat, output_shape, false);
        final_output->set_friendly_name(gather_name + "/output");

        // Replace the original Gather with the final Reshape
        ov::replace_node(gather, final_output);
        ov::copy_runtime_info(gather,
                              {reshaped_indices,
                               experts_start,
                               range_m_tiled,
                               new_indices,
                               flat_indices,
                               flat_weights,
                               gathered_flat,
                               final_output});

        std::cout << "  Transformed Gather: " << gather_name << " [" << N << "," << M << "," << K << "] with indices["
                  << I << "]" << std::endl;
        model_changed = true;
    }

    if (model_changed) {
        LOG_INFO("GatherTo2DGather: Transformed " << gathers_to_transform.size() << " Gather nodes");
    }

    return model_changed;
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
