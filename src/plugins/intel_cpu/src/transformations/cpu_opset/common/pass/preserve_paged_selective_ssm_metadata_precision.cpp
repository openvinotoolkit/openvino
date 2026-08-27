// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preserve_paged_selective_ssm_metadata_precision.hpp"

#include <cstddef>
#include <memory>
#include <unordered_set>
#include <vector>

#include "nodes/paged_selective_ssm.h"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_input.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "transformations/cpu_opset/common/pass/insert_convert_after_extension.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {
namespace {

using NodeList = std::vector<std::shared_ptr<ov::Node>>;
using NodeSet = std::unordered_set<const ov::Node*>;

bool is_metadata_port(size_t port_index) {
    if (port_index >= paged_ssm_input_count) {
        return false;
    }
    return is_paged_ssm_metadata_port(static_cast<PagedSelectiveSSMInputPort>(port_index));
}

bool is_paged_ssm_metadata_input(const ov::Input<ov::Node>& input) {
    return ov::is_type<ov::op::internal::PagedSelectiveSSM>(input.get_node()) && is_metadata_port(input.get_index());
}

NodeSet insert_extension_converts(const std::shared_ptr<ov::Model>& model) {
    NodeSet existing_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        existing_nodes.insert(node.get());
    }

    ov::pass::InsertConvertAfterExtension insert_convert(false);
    for (const auto& node : model->get_ordered_ops()) {
        insert_convert.apply(node);
    }

    NodeSet inserted_converts;
    for (const auto& node : model->get_ordered_ops()) {
        if (existing_nodes.count(node.get()) == 0 && ov::is_type<ov::op::v0::Convert>(node)) {
            inserted_converts.insert(node.get());
        }
    }
    return inserted_converts;
}

bool collect_i64_path(ov::Input<ov::Node> input,
                      const NodeSet& inserted_converts,
                      NodeList& protected_nodes,
                      NodeSet& protected_node_set) {
    auto source = input.get_source_output();
    bool changed = false;
    if (inserted_converts.count(source.get_node()) != 0 && source.get_element_type() == ov::element::i32) {
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(source.get_node_shared_ptr());
        if (convert && convert->get_input_element_type(0) == ov::element::i64) {
            source = convert->input_value(0);
            input.replace_source_output(source);
            changed = true;
        }
    }

    if (source.get_element_type() != ov::element::i64) {
        return changed;
    }

    const auto node = source.get_node_shared_ptr();
    if (!protected_node_set.insert(node.get()).second) {
        return changed;
    }
    protected_nodes.push_back(node);

    for (auto producer_input : node->inputs()) {
        changed = collect_i64_path(producer_input, inserted_converts, protected_nodes, protected_node_set) || changed;
    }
    return changed;
}

bool preserve_metadata_precision(const std::shared_ptr<ov::Model>& model) {
    const auto inserted_converts = insert_extension_converts(model);
    const auto ordered_ops = model->get_ordered_ops();

    bool changed = !inserted_converts.empty();
    NodeSet model_nodes;
    model_nodes.reserve(ordered_ops.size());
    NodeList protected_nodes;
    NodeSet protected_node_set;
    for (const auto& node : ordered_ops) {
        model_nodes.insert(node.get());
        const auto paged_ssm = ov::as_type_ptr<ov::op::internal::PagedSelectiveSSM>(node);
        if (paged_ssm) {
            for (const auto port : paged_ssm_metadata_ports) {
                changed = collect_i64_path(paged_ssm->input(input_port_index(port)),
                                           inserted_converts,
                                           protected_nodes,
                                           protected_node_set) ||
                          changed;
            }
        }
        const auto subgraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node);
        if (subgraph) {
            for (const auto& body : subgraph->get_functions()) {
                if (body) {
                    changed = preserve_metadata_precision(body) || changed;
                }
            }
        }
    }

    for (const auto& node : protected_nodes) {
        if (!ov::is_conversion_disabled(node, ov::element::i64, ov::element::i32)) {
            ov::disable_conversion(node, ov::element::i64, ov::element::i32);
            changed = true;
        }

        for (const auto& output : node->outputs()) {
            if (output.get_element_type() != ov::element::i64) {
                continue;
            }

            std::vector<ov::Input<ov::Node>> boundary_inputs;
            for (const auto& target_input : output.get_target_inputs()) {
                const auto* target_node = target_input.get_node();
                if (!model_nodes.count(target_node) || protected_node_set.count(target_node) ||
                    is_paged_ssm_metadata_input(target_input) || ov::is_type<ov::op::v0::Convert>(target_node) ||
                    ov::is_type<ov::op::v0::Result>(target_node)) {
                    continue;
                }
                boundary_inputs.push_back(target_input);
            }

            if (boundary_inputs.empty()) {
                continue;
            }

            const auto convert = std::make_shared<ov::op::v0::Convert>(output, ov::element::i32);
            ov::copy_runtime_info(node, convert);
            for (auto& input : boundary_inputs) {
                input.replace_source_output(convert);
            }
            changed = true;
        }
    }

    return changed;
}

}  // namespace

bool PreservePagedSelectiveSSMMetadataPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    return preserve_metadata_precision(model);
}

}  // namespace ov::intel_cpu
