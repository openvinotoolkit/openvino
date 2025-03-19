// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief File provides OpenVINO Runtime Execution Model Information
 * @file openvino/runtime/exec_model_info.hpp
 */

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/runtime/common.hpp"

/**
 * @brief A namespace with const values for Execution Graph parameters names.
 * @ingroup ov_dev_exec_model
 * Executable Model Info is represented in ov::Model format with general ExecutionNode nodes inside
 * including connections between the nodes. Each node describes an executable hardware-specific
 * primitive and stores its parameters within ExecutionNode::get_rt_info map.
 * There is a list of general keys for the parameters map.
 */
namespace ov {

namespace exec_model_info {
/**
 * @ingroup ov_dev_exec_model
 * @brief Used to get a string of layer names separated by a comma
 *        from the original IR, which were fused/merged to the current executable primitive.
 */
static const char ORIGINAL_NAMES[] = "originalLayersNames";

/**
 * @ingroup ov_dev_exec_model
 * @brief Used to get a type of the executable primitive.
 */
static const char IMPL_TYPE[] = "primitiveType";

/**
 * @ingroup ov_dev_exec_model
 * @brief Used to get output precisions of the executable primitive.
 */
static const char OUTPUT_PRECISIONS[] = "outputPrecisions";

/**
 * @ingroup ov_dev_exec_model
 * @brief Used to get a value of execution time of the executable primitive, where Mcs = microseconds (1μs=0.000001s).
 */
static const char PERF_COUNTER[] = "execTimeMcs";

/**
 * @ingroup ov_dev_exec_model
 * @brief Used to get output layouts of primitive.
 */
static const char OUTPUT_LAYOUTS[] = "outputLayouts";

/**
 * @ingroup ov_dev_exec_model
 * @brief Used to get an execution order of primitive.
 */
static const char EXECUTION_ORDER[] = "execOrder";

/**
 * @ingroup ov_dev_exec_model
 * @brief Used to get a type of primitive.
 */
static const char LAYER_TYPE[] = "layerType";

/**
 * @ingroup ov_dev_exec_model
 * @brief Used to get runtime precision of the executable primitive.
 */
static const char RUNTIME_PRECISION[] = "runtimePrecision";

/**
 * @ingroup ov_dev_exec_model
 * @brief The Execution node which is used to represent node in execution graph.
 *
 * It contains the following type of information in node runtime information:
 * - ExecGraphInfoSerialization::ORIGINAL_NAMES
 * - ExecGraphInfoSerialization::IMPL_TYPE
 * - ExecGraphInfoSerialization::OUTPUT_PRECISIONS
 * - ExecGraphInfoSerialization::PERF_COUNTER
 * - ExecGraphInfoSerialization::OUTPUT_LAYOUTS
 * - ExecGraphInfoSerialization::EXECUTION_ORDER
 * - ExecGraphInfoSerialization::LAYER_TYPE
 * - ExecGraphInfoSerialization::RUNTIME_PRECISION
 */
class OPENVINO_RUNTIME_API ExecutionNode : public ov::op::Op {
public:
    OPENVINO_OP("ExecutionNode");

    /**
     * A default constructor with no node inputs and 0 output ports.
     */
    ExecutionNode();

    /**
     * @brief      Constructs a new execution node with a given parameters
     *
     * @param[in]  arguments    Inputs nodes
     * @param[in]  output_size  A number of output ports
     */
    ExecutionNode(const ov::OutputVector& arguments, size_t output_size = 1);

    /**
     * @brief      Creates a new execution node with the same state, but different input nodes
     *
     * @param[in]  inputs  The input nodes
     *
     * @return     A newly created execution node
     */
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override;

    /**
     * @brief      Visits attributes of the node
     *
     * @param[in]  visitor  An attribute visitor
     *
     * @return     Returns `true` if an operation has completed successfully
     */
    bool visit_attributes(ov::AttributeVisitor& /*visitor*/) override;
};

}  // namespace exec_model_info
}  // namespace ov
