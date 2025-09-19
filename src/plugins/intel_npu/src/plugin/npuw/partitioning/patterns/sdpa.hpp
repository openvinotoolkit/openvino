// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

namespace online {
class Snapshot;  // Forward declaration
}  // namespace online

// Note: the patterns below are only utilized by the online partitioner
namespace patterns {
namespace attn {

struct AttentionParams {
    size_t batch_size = 0;
    size_t num_heads = 0;
    size_t sequence_length = 0;
    size_t head_dim = 0;
    ov::element::Type data_type = ov::element::dynamic;
    bool has_mask = false;

    // Original tensor shapes
    ov::Shape q_shape;
    ov::Shape k_shape;
    ov::Shape v_shape;
    ov::Shape output_shape;

    // Dimension indices for dynamic reshaping
    struct DimensionInfo {
        size_t batch_dim = 0;
        size_t heads_dim = 1;
        size_t sequence_dim = 2;
        size_t head_dim_idx = 3;
    };

    DimensionInfo q_dims;
    DimensionInfo k_dims;
    DimensionInfo v_dims;
};

// Function to extract attention parameters from SDPA pattern nodes
AttentionParams extractAttentionParamsFromSDPAPattern(const std::shared_ptr<ov::Node>& matmul1,
                                                      const std::shared_ptr<ov::Node>& matmul2,
                                                      const std::shared_ptr<ov::Node>& softmax_node,
                                                      const std::shared_ptr<ov::Node>& add_node);

// Function to analyze ov::Model and extract attention parameters with dimension mapping
AttentionParams identifyAttentionParamsFromModel(const std::shared_ptr<ov::Model>& model);

class SDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::SDPA");
    SDPA(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
