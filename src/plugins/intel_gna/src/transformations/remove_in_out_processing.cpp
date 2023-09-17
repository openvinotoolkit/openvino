// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/remove_in_out_processing.hpp"

#include "backend/gna_limitations.hpp"
#include "common/graph_utils.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov::opset12;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::limitations;

namespace {

inline bool is_preprocessing_layer_not_supported(std::shared_ptr<ov::Node>& layer) {
    // Gather layer is not supported by GNA and has to be executed on CPU
    if (std::dynamic_pointer_cast<ov::opset1::Gather>(layer) || std::dynamic_pointer_cast<ov::opset7::Gather>(layer) ||
        std::dynamic_pointer_cast<ov::opset8::Gather>(layer)) {
        return true;
    }

    // Verify that transpose layer cannot be executed on GNA
    if (std::dynamic_pointer_cast<ov::opset1::Transpose>(layer)) {
        return !Limitations::is_transpose_supported(layer);
    }

    return false;
}
/*
  Support only one input node as input 0
 */
inline std::shared_ptr<ov::Model> copy_single_input_node(std::shared_ptr<ov::Node> node) {
    const ov::element::Type& input_type = node->get_input_element_type(0);
    const ov::Shape& input_shape = node->get_input_shape(0);

    auto param = std::make_shared<Parameter>(input_type, input_shape);
    ov::OutputVector input_nodes = node->input_values();
    input_nodes[0] = param;
    auto node_copy = node->clone_with_new_inputs(input_nodes);
    auto result = std::make_shared<Result>(node_copy);
    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    return model;
}

inline bool is_skip_operation(const std::shared_ptr<ov::Node>& node) {
    return std::dynamic_pointer_cast<Reshape>(node) != nullptr;
}

}  // namespace

bool RemoveInputsProcessing::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RemoveInputsProcessing);
    bool result = false;

    for (const auto& param_node : model->inputs()) {
        for (auto& param_target : param_node.get_target_inputs()) {
            auto target_node = graph_utils::get_next_node_skipping_certain(param_target.get_node()->shared_from_this(),
                                                                           is_skip_operation);
            // Parameter -> Transpose, Parameter -> Gather
            if (is_preprocessing_layer_not_supported(target_node)) {
                if (m_input_subgraphs) {
                    m_input_subgraphs->emplace(param_node.get_node_shared_ptr()->get_friendly_name(),
                                               copy_single_input_node(target_node));
                }
                pass::helper::remove_single_input_node(target_node);
                result = true;
            }
        }
    }
    return result;
}

bool RemoveOutputsProcessing::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RemoveOutputsProcessing);
    bool result = false;
    for (std::shared_ptr<ov::Node> r_node : model->get_results()) {
        for (auto& r_input : r_node->input_values()) {
            auto r_input_node =
                graph_utils::get_prev_node_skipping_certain(r_input.get_node_shared_ptr(), is_skip_operation);
            // Transpose -> Result, Gather -> Result
            if (is_preprocessing_layer_not_supported(r_input_node)) {
                if (m_output_subgraphs) {
                    m_output_subgraphs->emplace(r_input.get_node_shared_ptr()->get_friendly_name(),
                                                copy_single_input_node(r_input_node));
                }
                pass::helper::remove_single_input_node(r_input_node);
                result = true;
            }
        }
    }
    return result;
}