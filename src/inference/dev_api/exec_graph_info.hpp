// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file defines names to be used by plugins to create execution graph.
 * It's an API between plugin and WorkBench tool.
 * @file exec_graph_info.hpp
 */

#pragma once

#include <string>

#include "openvino/op/op.hpp"
#include "openvino/runtime/exec_model_info.hpp"

/**
 * @brief A namespace with const values for Execution Graph parameters names.
 * @ingroup ie_dev_exec_graph
 * Executable Graph Info is represented in CNNNetwork format with general ExecutionNode nodes inside
 * including connections between the nodes. Each node describes an executable hardware-specific
 * primitive and stores its parameters within ExecutionNode::get_rt_info map.
 * There is a list of general keys for the parameters map.
 */
namespace ExecGraphInfoSerialization {

using ov::exec_model_info::EXECUTION_ORDER;
using ov::exec_model_info::ExecutionNode;
using ov::exec_model_info::IMPL_TYPE;
using ov::exec_model_info::LAYER_TYPE;
using ov::exec_model_info::ORIGINAL_NAMES;
using ov::exec_model_info::OUTPUT_LAYOUTS;
using ov::exec_model_info::OUTPUT_PRECISIONS;
using ov::exec_model_info::PERF_COUNTER;
using ov::exec_model_info::RUNTIME_PRECISION;

}  // namespace ExecGraphInfoSerialization
