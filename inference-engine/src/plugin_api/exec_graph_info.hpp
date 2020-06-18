// Copyright (C) 2018-2020 Intel Corporation
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
#include <ngraph/function.hpp>

/**
 * @brief A namespace with const values for Execution Graph parameters names.
 *  
 *        Executable Graph Info is represented in ICNNNetwork format with general CNNLayer nodes inside
 *        including connections between the nodes. Each node describes an executable hardware-specific
 *        primitive and stores its parameters within CNNLayer::params map.
 *        There is a list of general keys for the parameters map.
 */
namespace ExecGraphInfoSerialization {

/**
 * @brief Used to get a string of layer names separated by a comma
 *        from the original IR, which were fused/merged to the current executable primitive.
 */
static const char ORIGINAL_NAMES[] = "originalLayersNames";

/**
 * @brief Used to get a type of the executable primitive.
 */
static const char IMPL_TYPE[] = "primitiveType";

/**
 * @brief Used to get output precisions of the executable primitive.
 */
static const char OUTPUT_PRECISIONS[] = "outputPrecisions";

/**
 * @brief Used to get a value of execution time of the executable primitive.
 */
static const char PERF_COUNTER[] = "execTimeMcs";

/**
 * @brief Used to get output layouts of primitive.
 */
static const char OUTPUT_LAYOUTS[] = "outputLayouts";

/**
 * @brief Used to get an execution order of primitive.
 */
static const char EXECUTION_ORDER[] = "execOrder";

/**
 * @brief Used to get a type of primitive.
 */
static const char LAYER_TYPE[] = "layerType";

class INFERENCE_ENGINE_API_CLASS(ExecutionNode) : public ngraph::Node {
public:
    static constexpr ngraph::NodeTypeInfo type_info { "ExecutionNode", 0 };
    const ngraph::NodeTypeInfo& get_type_info() const override;

    ExecutionNode() = default;

    ExecutionNode(const ngraph::OutputVector& arguments, size_t output_size = 1) :
        Node(arguments, output_size) { }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
        auto cloned = std::make_shared<ExecutionNode>();

        cloned->set_arguments(inputs);

        for (auto kvp : get_rt_info())
            cloned->get_rt_info()[kvp.first] = kvp.second;

        for (size_t i = 0; i < get_output_size(); ++i)
            cloned->set_output_type(i, get_output_element_type(i), get_output_partial_shape(i));

        return cloned;
    }
};

}  // namespace ExecGraphInfoSerialization
