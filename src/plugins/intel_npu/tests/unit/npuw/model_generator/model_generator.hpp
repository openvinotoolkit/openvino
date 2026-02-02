// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

class ModelGenerator {
public:
    ModelGenerator() = default;

    std::shared_ptr<ov::Model> get_model_with_one_op();
    std::shared_ptr<ov::Model> get_model_without_repeated_blocks();
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks(std::size_t repetitions);
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks();

    // Build model with repeating blocks and configurable ov::Result consumers:
    //   - repetitions: number of repeating blocks
    //   - block_indices: vector of block indices (0-based) that should have ov::Result consumers
    //                empty vector means no additional Results, only the final tail Result
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_and_results(
        std::size_t repetitions,
        const std::vector<std::size_t>& block_indices);

    // Build model with repeating blocks where selected blocks accept external ov::Parameter inputs.
    //   - repetitions: number of repeating blocks (minimum 1)
    //   - block_indices: vector of block indices (0-based) that should consume a distinct ov::Parameter as one of the
    //                    first op inputs; indices outside [0, repetitions) are ignored
    std::shared_ptr<ov::Model> get_model_with_repeated_blocks_and_parameters(
        std::size_t repetitions,
        const std::vector<std::size_t>& block_indices);

    // Build model with repeating blocks where the final op in each block has multiple outputs (TopK values + indices).
    //   - repetitions: number of repeating blocks
    //   - last_block_has_direct_result:
    //       Option1 (false): for all blocks, multi-output node feeds only the next block; last block feeds only the
    //       tail Option2 (true): same as above, plus the last block also feeds a direct ov::Result from one of its
    //       outputs
    std::shared_ptr<ov::Model> get_model_with_multi_output_repeating_blocks(std::size_t repetitions,
                                                                            bool last_block_has_direct_result);

private:
    std::shared_ptr<ov::Node> get_block(const std::shared_ptr<ov::Node>& input);
    void set_name(const std::shared_ptr<ov::Node>& node);

private:
    std::vector<std::shared_ptr<ov::Node>> m_nodes;
    size_t m_name_idx;
};
