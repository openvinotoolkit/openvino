// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "onnx_import/core/node.hpp"
#include "openvino/core/deprecated.hpp"

namespace ngraph {
namespace onnx_import {
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
    using container_type = std::map<OpInput, Output<ngraph::Node>>;

    OPENVINO_SUPPRESS_DEPRECATED_START
    explicit OpInputMap(const onnx_import::Node& node, std::size_t gates_count);
    OPENVINO_SUPPRESS_DEPRECATED_END
    OpInputMap(container_type&& map);
    virtual ~OpInputMap() = default;

    Output<ngraph::Node>& at(const OpInput& key);
    const Output<ngraph::Node>& at(const OpInput& key) const;

    container_type m_map;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

///
/// \brief      This structure aggregates operator's attributes.
///
struct OpAttributes {
    OPENVINO_SUPPRESS_DEPRECATED_START
    explicit OpAttributes(const Node& node);
    OPENVINO_SUPPRESS_DEPRECATED_END
    virtual ~OpAttributes() = default;

    ngraph::op::RecurrentSequenceDirection m_direction;
    std::int64_t m_hidden_size;
    float m_clip_threshold;
    std::vector<std::string> m_activations;
    std::vector<float> m_activations_alpha;
    std::vector<float> m_activations_beta;
};

}  // namespace recurrent
}  // namespace onnx_import
}  // namespace ngraph
