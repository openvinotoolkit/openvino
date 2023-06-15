// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/mark_output_sinking.hpp"

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
#include "transformations/rt_info/transpose_sinking_attr.hpp"

using namespace ov::opset12;
using namespace ov::intel_gna::pass;

namespace {
inline bool is_skip_operation(const std::shared_ptr<ov::Node>& node) {
    return std::dynamic_pointer_cast<Reshape>(node) != nullptr;
}
} // namespace

bool MarkOutputSinking::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(MarkOutputSinking);
    bool result = false;
    for (std::shared_ptr<ov::Node> r_node : model->get_results()) {
        for (auto& r_input : r_node->input_values()) {
            auto r_input_node =
                graph_utils::get_prev_node_skipping_certain(r_input.get_node_shared_ptr(), is_skip_operation);
            // Transpose -> Result, Gather -> Result
            if (!std::dynamic_pointer_cast<ov::opset1::Gather>(r_input_node) && !std::dynamic_pointer_cast<ov::opset7::Gather>(r_input_node) ||
                !std::dynamic_pointer_cast<ov::opset8::Gather>(r_input_node) && !std::dynamic_pointer_cast<ov::opset1::Transpose>(r_input_node)) {
                continue;
            }
            result = true;
            mark_as_no_sinking_node(r_input_node);
        }
    }
    return result;
}
