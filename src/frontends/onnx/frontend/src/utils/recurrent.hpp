// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>

#include "core/node.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace recurrent {
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

///
/// \brief      This class describes a recurrent operation input name
///
enum class OpInput {
    X,            // Packed input sequences.
                  // Shape: [seq_length, batch_size, input_size]
    W,            // Weight tensor for the gates.
                  // Shape: [num_directions, gates_count*hidden_size, input_size]
    R,            // The recurrence weight tensor.
                  // Shape: [num_directions, gates_count*hidden_size, hidden_size]
    B,            // The bias tensor for gates.
                  // Shape [num_directions, gates_count*hidden_size]
    SEQ_LENGTHS,  // The lengths of the sequences in a batch. Shape [batch_size]
    INIT_H,       // The initial value of the hidden.
                  // Shape [num_directions, batch_size, hidden_size]
};

///
/// \brief      This structure aggregates operator's inptus in a key-value map.
///
struct OpInputMap {
    using container_type = std::map<OpInput, ov::Output<ov::Node>>;

    explicit OpInputMap(const ov::frontend::onnx::Node& node, std::size_t gates_count);

    OpInputMap(container_type&& map);
    virtual ~OpInputMap() = default;

    ov::Output<ov::Node>& at(const OpInput& key);
    const ov::Output<ov::Node>& at(const OpInput& key) const;

    container_type m_map;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

///
/// \brief      This structure aggregates operator's attributes.
///
struct OpAttributes {
    explicit OpAttributes(const Node& node);

    virtual ~OpAttributes() = default;

    ov::op::RecurrentSequenceDirection m_direction;
    std::int64_t m_hidden_size;
    float m_clip_threshold;
    std::vector<std::string> m_activations;
    std::vector<float> m_activations_alpha;
    std::vector<float> m_activations_beta;
};

}  // namespace recurrent
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
