// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace util {

/**
 * @brief Extracts a subgraph bounded by the given input and output ports as a standalone model.
 * The original model is not modified; the extracted model is a deep clone.
 *
 * @param model            Source model.
 * @param subgraph_inputs  Input ports that form the subgraph boundary. Each port is replaced
 *                         by a new Parameter whose element type and partial shape match the port.
 * @param subgraph_outputs Output ports that define the subgraph results.
 * @return A new ov::Model representing the extracted subgraph.
 */
TRANSFORMATIONS_API std::shared_ptr<ov::Model> extract_subgraph(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<ov::Input<ov::Node>>& subgraph_inputs,
    const std::vector<ov::Output<ov::Node>>& subgraph_outputs);

/**
 * @brief Extracts a subgraph by resolving boundary ports from node friendly names.
 *
 * Convenience wrapper over the port-based overload. Each multimap entry maps a node
 * friendly name to a port index; multiple entries with the same name select different
 * ports of the same node. Throws ov::Exception if any name is not found in the model.
 *
 * @param model             Source model.
 * @param subgraph_inputs   Map of { friendly_name -> input_port_index } for boundary inputs.
 * @param subgraph_outputs  Map of { friendly_name -> output_port_index } for boundary outputs.
 * @return A new ov::Model representing the extracted subgraph.
 */
TRANSFORMATIONS_API std::shared_ptr<ov::Model> extract_subgraph(
    const std::shared_ptr<ov::Model>& model,
    const std::multimap<std::string, size_t>& subgraph_inputs,
    const std::multimap<std::string, size_t>& subgraph_outputs);

}  // namespace util
}  // namespace op
}  // namespace ov
