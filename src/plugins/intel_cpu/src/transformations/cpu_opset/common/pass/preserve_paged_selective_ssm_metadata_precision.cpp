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

void collect_i64_path(const ov::Output<ov::Node>& output, NodeList& path_nodes, NodeSet& path_node_set) {
    if (output.get_element_type() != ov::element::i64) {
        return;
    }

    const auto node = output.get_node_shared_ptr();
    if (!path_node_set.insert(node.get()).second) {
        return;
    }
    path_nodes.push_back(node);
    for (const auto& input : node->inputs()) {
        collect_i64_path(input.get_source_output(), path_nodes, path_node_set);
    }
}

bool add_i32_boundaries(const std::shared_ptr<ov::Node>& node, const NodeSet& path_nodes) {
    bool changed = false;
    for (const auto& output : node->outputs()) {
        if (output.get_element_type() != ov::element::i64 && output.get_element_type() != ov::element::u64) {
            continue;
        }

        std::vector<ov::Input<ov::Node>> consumers;
        for (const auto& input : output.get_target_inputs()) {
            const auto* consumer = input.get_node();
            if (path_nodes.count(consumer) == 0 && !is_paged_ssm_metadata_input(input) &&
                !ov::is_type<ov::op::v0::Convert>(consumer) && !ov::is_type<ov::op::v0::Result>(consumer)) {
                consumers.push_back(input);
            }
        }
        if (consumers.empty()) {
            continue;
        }

        const auto convert = std::make_shared<ov::op::v0::Convert>(output, ov::element::i32);
        ov::copy_runtime_info(node, convert);
        for (auto& input : consumers) {
            input.replace_source_output(convert);
        }
        changed = true;
    }
    return changed;
}

bool preserve_metadata_precision(const std::shared_ptr<ov::Model>& model) {
    const auto ordered_ops = model->get_ordered_ops();
    NodeList path_nodes;
    NodeSet path_node_set;
    for (const auto& node : ordered_ops) {
        const auto paged_ssm = ov::as_type_ptr<ov::op::internal::PagedSelectiveSSM>(node);
        if (!paged_ssm) {
            continue;
        }
        for (const auto port : paged_ssm_metadata_ports) {
            collect_i64_path(paged_ssm->input_value(input_port_index(port)), path_nodes, path_node_set);
        }
    }

    bool changed = false;
    for (const auto& node : path_nodes) {
        if (!ov::is_conversion_disabled(node, ov::element::i64, ov::element::i32)) {
            ov::disable_conversion(node, ov::element::i64, ov::element::i32);
            changed = true;
        }
        changed = add_i32_boundaries(node, path_node_set) || changed;
    }

    ov::pass::InsertConvertAfterExtension insert_convert(false);
    for (const auto& node : ordered_ops) {
        if (path_node_set.count(node.get()) == 0) {
            changed = insert_convert.apply(node) || changed;
        }
    }

    for (const auto& node : ordered_ops) {
        const auto subgraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node);
        if (!subgraph) {
            continue;
        }
        for (const auto& body : subgraph->get_functions()) {
            if (body) {
                changed = preserve_metadata_precision(body) || changed;
            }
        }
    }
    return changed;
}

}  // namespace

bool PreservePagedSelectiveSSMMetadataPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    return preserve_metadata_precision(model);
}

}  // namespace ov::intel_cpu
