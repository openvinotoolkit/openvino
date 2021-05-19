// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file defines names to be used by plugins to create execution graph.
 * It's an API between plugin and WorkBench tool.
 * @file exec_graph_info.hpp
 */

#pragma once

#include <ie_api.h>
#include <ie_parameter.hpp>
#include <string>

#include <ngraph/node.hpp>

/**
 * @brief A namespace with const values for Execution Graph parameters names.
 *
 * Executable Graph Info is represented in CNNNetwork format with general ExecutionNode nodes inside
 * including connections between the nodes. Each node describes an executable hardware-specific
 * primitive and stores its parameters within ExecutionNode::get_rt_info map.
 * There is a list of general keys for the parameters map.
 */
namespace ExecGraphInfoSerialization {

/**
 * @ingroup ie_dev_exec_graph
 * @brief Used to get a string of layer names separated by a comma
 *        from the original IR, which were fused/merged to the current executable primitive.
 */
static const char ORIGINAL_NAMES[] = "originalLayersNames";

/**
 * @ingroup ie_dev_exec_graph
 * @brief Used to get a type of the executable primitive.
 */
static const char IMPL_TYPE[] = "primitiveType";

/**
 * @ingroup ie_dev_exec_graph
 * @brief Used to get output precisions of the executable primitive.
 */
static const char OUTPUT_PRECISIONS[] = "outputPrecisions";

/**
 * @ingroup ie_dev_exec_graph
 * @brief Used to get a value of execution time of the executable primitive.
 */
static const char PERF_COUNTER[] = "execTimeMcs";

/**
 * @ingroup ie_dev_exec_graph
 * @brief Used to get output layouts of primitive.
 */
static const char OUTPUT_LAYOUTS[] = "outputLayouts";

/**
 * @ingroup ie_dev_exec_graph
 * @brief Used to get an execution order of primitive.
 */
static const char EXECUTION_ORDER[] = "execOrder";

/**
 * @ingroup ie_dev_exec_graph
 * @brief Used to get a type of primitive.
 */
static const char LAYER_TYPE[] = "layerType";

/**
 * @ingroup ie_dev_exec_graph
 * @brief Used to get runtime precision of the executable primitive.
 */
static const char RUNTIME_PRECISION[] = "runtimePrecision";

/**
 * @ingroup ie_dev_exec_graph
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
class INFERENCE_ENGINE_API_CLASS(ExecutionNode) : public ngraph::Node {
public:
    static constexpr ngraph::NodeTypeInfo type_info { "ExecutionNode", 0 };
    const ngraph::NodeTypeInfo& get_type_info() const override;

    /**
     * A default constructor with no node inputs and 0 output ports.
     */
    ExecutionNode() = default;

    /**
     * @brief      Constructs a new execution node with a given parameters
     *
     * @param[in]  arguments    Inputs nodes
     * @param[in]  output_size  A number of output ports
     */
    ExecutionNode(const ngraph::OutputVector& arguments, size_t output_size = 1) :
        Node(arguments, output_size) { }

    /**
     * @brief      Creates a new execution node with the same state, but different input nodes
     *
     * @param[in]  inputs  The input nodes
     *
     * @return     A newly created execution node
     */
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
        auto cloned = std::make_shared<ExecutionNode>();

        cloned->set_arguments(inputs);

        for (auto kvp : get_rt_info())
            cloned->get_rt_info()[kvp.first] = kvp.second;

        for (size_t i = 0; i < get_output_size(); ++i)
            cloned->set_output_type(i, get_output_element_type(i), get_output_partial_shape(i));

        return cloned;
    }

    /**
     * @brief      Visits attributes of the node
     *
     * @param[in]  visitor  An attribute visitor
     *
     * @return     Returns `true` if an operation has completed successfully
     */
    bool visit_attributes(ngraph::AttributeVisitor& /*visitor*/) override {
        return true;
    }
};

}  // namespace ExecGraphInfoSerialization
