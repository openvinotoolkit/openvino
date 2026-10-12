// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "force_fp32_selective.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef GPU_DEBUG_CONFIG

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/variable_extension.hpp"
#include "openvino/pass/manager.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "plugin/transformations/utils.hpp"

namespace ov {
namespace intel_gpu {

namespace {

// Update explicit precision attributes as well as the inferred output descriptor.
// Otherwise validation can restore an operation's original f16 output type.
class FP32AttributeVisitor : public ov::AttributeVisitor {
public:
    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (name == "output_type" || name == "destination_type" || name == "out_type") {
            if (auto* type = ov::as_type<ov::AttributeAdapter<ov::element::Type>>(&adapter)) {
                if (type->get() == "f16")
                    type->set("f32");
            } else if (auto* type = ov::as_type<ov::AttributeAdapter<ov::element::Type_t>>(&adapter)) {
                if (type->get() == "f16")
                    type->set("f32");
            }
        } else if (name == "input_data_types" || name == "output_data_types") {
            if (auto* types = ov::as_type<ov::AttributeAdapter<ov::element::TypeVector>>(&adapter)) {
                auto value = types->get();
                std::replace(value.begin(), value.end(), ov::element::f16, ov::element::f32);
                types->set(value);
            }
        }
    }

    // Do not recurse into nested attributes or subgraphs.
    void on_adapter(const std::string&, ov::VisitorAdapter&) override {}
};

void promote_precision_attributes(const std::shared_ptr<ov::Node>& node) {
    // Constants, model boundaries, and shared state require coordinated conversion.
    if (ov::is_type<ov::op::v0::Constant>(node) || ov::is_type<ov::op::v0::Parameter>(node) ||
        ov::is_type<ov::op::v0::Result>(node) || ov::is_type<ov::op::util::MultiSubGraphOp>(node) ||
        dynamic_cast<ov::op::util::VariableExtension*>(node.get())) {
        return;
    }

    FP32AttributeVisitor visitor;
    node->visit_attributes(visitor);
}

// Pass-through ops: no arithmetic, just data movement.
// These are automatically absorbed into f32 regions when connected to f32 ops,
// preventing f32->f16->f32 roundtrip truncation.
bool is_passthrough_op(const std::string& type_name) {
    static const std::unordered_set<std::string> passthrough_types = {
        "Reshape", "Transpose", "Squeeze", "Unsqueeze",
        "StridedSlice", "Gather", "Concat", "Split", "VariadicSplit",
        "ShapeOf", "Broadcast"
    };
    return passthrough_types.count(type_name) > 0;
}

bool matches_type(const std::vector<std::string>& forced_types, const std::string& type_name) {
    return std::find(forced_types.begin(), forced_types.end(), type_name) != forced_types.end();
}

bool matches_name(const std::vector<std::string>& forced_names, const std::string& node_name) {
    for (const auto& pattern : forced_names) {
        if (node_name.find(pattern) != std::string::npos)
            return true;
    }
    return false;
}

bool has_fp16_output(const std::shared_ptr<ov::Node>& node) {
    for (const auto& output : node->outputs()) {
        if (output.get_element_type() == ov::element::f16)
            return true;
    }
    return false;
}

using OriginalOutputTypes = std::unordered_map<ov::Node*, ov::element::TypeVector>;

OriginalOutputTypes record_output_types(const std::shared_ptr<ov::Model>& model,
                                       const std::unordered_set<ov::Node*>& f32_region) {
    OriginalOutputTypes original_types;
    for (const auto& node : model->get_ops()) {
        if (!f32_region.count(node.get()))
            continue;
        auto& types = original_types[node.get()];
        for (const auto& output : node->outputs())
            types.push_back(output.get_element_type());
    }
    return original_types;
}

bool is_fp32_region_output(const ov::Output<ov::Node>& output, const OriginalOutputTypes& original_types) {
    const auto it = original_types.find(output.get_node());
    if (it == original_types.end())
        return false;
    const auto type = it->second.at(output.get_index());
    return type == ov::element::f16 || type == ov::element::f32;
}

// Phase 1: Seed Collection — identify ops that match the user filter
std::unordered_set<ov::Node*> collect_seeds(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<std::string>& forced_types,
    const std::vector<std::string>& forced_names) {
    const bool has_types = !forced_types.empty();
    const bool has_names = !forced_names.empty();
    std::unordered_set<ov::Node*> f32_region;

    for (const auto& node : model->get_ops()) {
        const std::string node_type = node->get_type_name();
        const std::string node_name = node->get_friendly_name();

        if (has_types && !matches_type(forced_types, node_type))
            continue;
        if (has_names && !matches_name(forced_names, node_name))
            continue;

        // GPU Placeholder represents an absent optional input, not a computation.
        if (ov::is_type<ov::intel_gpu::op::Placeholder>(node))
            continue;

        // Model boundaries keep their original types; outputless nodes are not seeds.
        if (ov::is_type<ov::op::v0::Parameter>(node) || ov::is_type<ov::op::v0::Result>(node) ||
            !has_fp16_output(node)) {
            continue;
        }

        f32_region.insert(node.get());
    }

    GPU_DEBUG_LOG << "[ForceFP32Selective] seeds: " << f32_region.size() << std::endl;
    return f32_region;
}

// Phase 2: Region Expansion — absorb pass-through ops connected to the f32
// region via flood-fill, preventing f32->f16->f32 roundtrip truncation.
//
// NOTE: We intentionally use one-side connectivity (OR), not bridge condition
// (AND). Requiring BOTH sides causes deadlock on multi-node passthrough chains:
//   [f32 A] -> Reshape -> Transpose -> [f32 B]
// With AND: neither is ever absorbed regardless of iteration count.
void expand_region(const std::shared_ptr<ov::Model>& model,
                   std::unordered_set<ov::Node*>& f32_region) {
    bool expanded = true;
    int expansion_rounds = 0;
    while (expanded) {
        expanded = false;
        expansion_rounds++;
        for (const auto& node : model->get_ops()) {
            if (f32_region.count(node.get()))
                continue;

            const std::string node_type = node->get_type_name();
            if (!is_passthrough_op(node_type))
                continue;

            if (!has_fp16_output(node))
                continue;

            // Check: does this node have at least one f32-region neighbor?
            bool has_f32_neighbor = false;
            for (const auto& input : node->inputs()) {
                auto* producer = input.get_source_output().get_node();
                if (f32_region.count(producer)) {
                    has_f32_neighbor = true;
                    break;
                }
            }
            if (!has_f32_neighbor) {
                for (const auto& output : node->outputs()) {
                    for (const auto& target : output.get_target_inputs()) {
                        if (f32_region.count(target.get_node())) {
                            has_f32_neighbor = true;
                            break;
                        }
                    }
                    if (has_f32_neighbor)
                        break;
                }
            }
            if (!has_f32_neighbor)
                continue;

            f32_region.insert(node.get());
            expanded = true;
        }
    }

    GPU_DEBUG_LOG << "[ForceFP32Selective] region after expansion: " << f32_region.size()
                  << " (rounds=" << expansion_rounds << ")" << std::endl;
}

// Bypass FP16/FP32 Convert chains from promoted or already-FP32 producers.
ov::Output<ov::Node> find_fp32_source(ov::Output<ov::Node> source,
                                    const OriginalOutputTypes& original_types) {
    const auto original_source = source;
    while (!original_types.count(source.get_node())) {
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(source.get_node_shared_ptr());
        if (!convert)
            break;

        const auto source_type = convert->get_input_element_type(0);
        const auto destination_type = convert->get_destination_type();
        if ((source_type != ov::element::f16 && source_type != ov::element::f32) ||
            (destination_type != ov::element::f16 && destination_type != ov::element::f32)) {
            return original_source;
        }
        source = convert->input_value(0);
    }

    if (!is_fp32_region_output(source, original_types) && source.get_element_type() != ov::element::f32)
        return original_source;

    return source;
}

// Phase 3a: Input Promotion — insert f16->f32 boundary Converts at region
// entry points and force output type to f32.
std::pair<bool, size_t> promote_inputs(
    const std::shared_ptr<ov::Model>& model,
    const OriginalOutputTypes& original_types) {
    bool changed = false;
    size_t global_convert_idx = 0;
    const auto desired_et = ov::element::f32;

    const auto ordered_ops = model->get_ordered_ops();

    // Pass 1: Connect region nodes directly; convert only external f16 inputs.
    for (const auto& node : ordered_ops) {
        if (!original_types.count(node.get()))
            continue;

        for (const auto& input : node->inputs()) {
            const auto incoming_output = input.get_source_output();
            const auto incoming_node = incoming_output.get_node_shared_ptr();
            // An absent optional input has no tensor to convert.
            if (ov::is_type<ov::intel_gpu::op::Placeholder>(incoming_node))
                continue;

            const auto input_et = incoming_output.get_element_type();
            if (input_et != ov::element::f16 && input_et != desired_et)
                continue;

            const auto fp32_source = find_fp32_source(incoming_output, original_types);
            if (is_fp32_region_output(fp32_source, original_types) ||
                fp32_source.get_element_type() == desired_et) {
                if (fp32_source != incoming_output)
                    input.replace_source_output(fp32_source);
                continue;
            }

            // Change this edge only, preserving external Converts and their users.
            auto convert = std::make_shared<ov::op::v0::Convert>(incoming_output, desired_et);
            convert->set_friendly_name(
                incoming_node->get_friendly_name() + "_increase_precision_" +
                std::to_string(global_convert_idx));
            ov::copy_runtime_info(incoming_node, convert);
            input.replace_source_output(convert);
            global_convert_idx++;
        }
    }

    // Pass 2: Update explicit precision attributes and force FP16 outputs to f32.
    for (const auto& node : ordered_ops) {
        if (!original_types.count(node.get()))
            continue;

        promote_precision_attributes(node);
        const auto& types = original_types.at(node.get());
        for (size_t port = 0; port < types.size(); port++) {
            if (types[port] == ov::element::f16)
                node->set_output_type(port, desired_et, node->get_output_partial_shape(port));
        }

        changed = true;
    }

    return {changed, global_convert_idx};
}

// Phase 3b: Output Restore — insert f32->f16 Converts at region exit points.
// Done separately after all promotions so that region-internal edges are never
// accidentally restored.
void restore_outputs(const std::shared_ptr<ov::Model>& model,
                     const OriginalOutputTypes& original_types,
                     size_t& global_convert_idx) {
    const auto desired_et = ov::element::f32;

    for (const auto& node : model->get_ops()) {
        if (!original_types.count(node.get()))
            continue;

        const auto& types = original_types.at(node.get());
        for (const auto& output : node->outputs()) {
            const auto original_type = types.at(output.get_index());
            if (original_type != ov::element::f16 || output.get_element_type() != desired_et)
                continue;
            std::vector<ov::Input<ov::Node>> outside_consumers;
            for (const auto& target_input : output.get_target_inputs()) {
                if (!original_types.count(target_input.get_node())) {
                    outside_consumers.push_back(target_input);
                }
            }
            for (auto& target_input : outside_consumers) {
                const auto existing_convert = ov::as_type<ov::op::v0::Convert>(target_input.get_node());
                if (existing_convert && existing_convert->get_destination_type() == original_type)
                    continue;

                auto convert = std::make_shared<ov::op::v0::Convert>(output, original_type);
                convert->set_friendly_name(
                    target_input.get_node()->shared_from_this()->get_friendly_name() +
                    "_restore_precision_" + std::to_string(global_convert_idx));
                ov::copy_runtime_info(node, convert);
                if (ov::is_type<ov::op::v0::Result>(target_input.get_node()))
                    convert->output(0).set_names(output.get_names());
                target_input.replace_source_output(convert);
                global_convert_idx++;
            }
        }
    }
}

// Report graph output transitions, excluding nodes disconnected by edge rewiring.
// This log does not describe the precision of the eventual GPU kernel.
void log_promoted_outputs(const std::shared_ptr<ov::Model>& model,
                          const OriginalOutputTypes& original_types) {
    if (ExecutionConfig::get_verbose() < static_cast<int>(LogLevel::LOG))
        return;

    size_t forced_count = 0;
    size_t forced_output_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        const auto it = original_types.find(node.get());
        if (it == original_types.end())
            continue;

        bool node_promoted = false;
        for (const auto& output : node->outputs()) {
            const auto port = output.get_index();
            if (it->second.at(port) != ov::element::f16 || output.get_element_type() != ov::element::f32)
                continue;

            node_promoted = true;
            forced_output_count++;
            GPU_DEBUG_LOG << "[ForceFP32Selective] FORCED: type=" << node->get_type_name()
                          << "  name=" << node->get_friendly_name()
                          << "  output=" << port << " f16->f32" << std::endl;
        }
        if (node_promoted)
            forced_count++;
    }

    GPU_DEBUG_LOG << "[ForceFP32Selective] Done. Region=" << original_types.size()
                  << "  Forced=" << forced_count << "  Outputs=" << forced_output_count << std::endl;
}

}  // namespace

bool ForceFP32Selective::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (m_forced_types.empty() && m_forced_names.empty())
        return false;

    // Phase 1: Seed Collection
    auto f32_region = collect_seeds(model, m_forced_types, m_forced_names);
    if (f32_region.empty())
        return false;

    // Phase 2: Region Expansion
    expand_region(model, f32_region);

    const auto original_types = record_output_types(model, f32_region);

    // Phase 3a: Input Promotion
    auto [changed, convert_idx] = promote_inputs(model, original_types);

    // Phase 3b: Output Restore
    restore_outputs(model, original_types, convert_idx);

    log_promoted_outputs(model, original_types);

    return changed;
}

}  // namespace intel_gpu
}  // namespace ov

#endif // GPU_DEBUG_CONFIG
