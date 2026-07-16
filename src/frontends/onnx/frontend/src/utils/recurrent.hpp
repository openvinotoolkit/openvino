// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "core/node.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace recurrent {

// Normalize a recurrent-operator input to target_rank. Dynamic rank is returned unchanged.
// rank > target: squeeze leading size-1 dims (rejects statically-known non-1 leading dims).
// rank == target-1: unsqueeze a leading dim (handles models that omit num_directions=1).
// Larger rank deficiency is rejected. op_name/input_name are used in error messages.
ov::Output<ov::Node> normalize_tensor_rank(const ov::Output<ov::Node>& input,
                                           int64_t target_rank,
                                           const std::string& op_name,
                                           const std::string& input_name);

// Runtime dimension values extracted from OV-layout X [batch, seq, input]
// and R [num_dir, gates*hidden, hidden]. Each member is a rank-1 i64 node.
struct LSTMDimensions {
    LSTMDimensions(const ov::Output<ov::Node>& x_ov_layout, const ov::Output<ov::Node>& r_ov_layout);

    ov::Output<ov::Node> batch_size;
    ov::Output<ov::Node> seq_length;
    ov::Output<ov::Node> num_directions;
    ov::Output<ov::Node> hidden_size;
};

// Default values for optional LSTM inputs (element_type typically matches X):
// default_bias:          zeros [num_directions, gates_count * hidden_size]
// default_sequence_lens: seq_length broadcast to [batch_size]
// default_initial_state: zeros [batch_size, num_directions, hidden_size]
ov::Output<ov::Node> default_bias(const LSTMDimensions& dims,
                                  const ov::element::Type& element_type,
                                  int64_t gates_count);
ov::Output<ov::Node> default_sequence_lens(const LSTMDimensions& dims);
ov::Output<ov::Node> default_initial_state(const LSTMDimensions& dims, const ov::element::Type& element_type);

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
