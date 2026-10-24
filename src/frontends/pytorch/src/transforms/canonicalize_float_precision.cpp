// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "canonicalize_float_precision.hpp"

#include <map>
#include <utility>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/util/log.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace ov::frontend::pytorch::pass {

namespace {
namespace v0 = ov::op::v0;

// Narrow Constants up to this many elements (scalars, eps, rotary tables) are folded to f32 instead of
// kept behind a decompression Convert. A size heuristic, not part of the decompression form itself:
// the f32 copy of such a Constant costs next to nothing.
constexpr size_t fold_to_f32_max_elements = 1024;

bool is_narrow(const element::Type& type) {
    return type == element::bf16 || type == element::f16;
}

bool is_side_channel(const std::shared_ptr<v0::Parameter>& param) {
    return param->get_friendly_name().rfind("__pa__", 0) == 0;
}
}  // namespace

bool CanonicalizeFloatPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // A body keeps the types it was converted with, so retyping the outer graph would feed f32 into bf16
    // bodies and fail validation. Leave such models as they are.
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::util::MultiSubGraphOp>(node)) {
            OPENVINO_DEBUG("CanonicalizeFloatPrecision skipped: the model has subgraph ops");
            return false;
        }
    }

    std::vector<element::Type> result_types;
    for (const auto& result : model->get_results()) {
        result_types.push_back(result->get_input_element_type(0));
    }

    // Weights and the model's inputs keep their narrow type, so ConvertPrecision must leave them alone.
    // Their consumers read them through a Convert to the same type: the model stays valid for
    // ConvertPrecision, which then retargets that Convert to f32 like any other narrow Convert.
    std::vector<std::shared_ptr<Node>> kept;
    const auto read_through_convert = [](const Output<Node>& output) {
        std::vector<Input<Node>> consumers;
        for (const auto& input : output.get_target_inputs()) {
            const auto consumer = input.get_node();
            // A Result keeps reading the node directly, so the output keeps its type and tensor names.
            if (ov::is_type<v0::Result>(consumer)) {
                continue;
            }
            const auto convert = ov::as_type<v0::Convert>(consumer);
            if (!convert || convert->get_destination_type() != element::f32) {
                consumers.push_back(input);
            }
        }
        if (consumers.empty()) {
            return false;
        }
        const auto convert = std::make_shared<v0::Convert>(output, output.get_element_type());
        for (auto& consumer : consumers) {
            consumer.replace_source_output(convert);
        }
        return true;
    };
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto constant = ov::as_type_ptr<v0::Constant>(node)) {
            if (!is_narrow(constant->get_element_type())) {
                continue;
            }
            // Small Constants (scalars, eps, rotary tables) are left to ConvertPrecision, which folds them
            // to f32; one that is only read by a Convert(f32) or a Result is already in its final form.
            const auto& targets = constant->output(0).get_target_inputs();
            const bool final_form = std::all_of(targets.begin(), targets.end(), [](const Input<Node>& target) {
                const auto convert = ov::as_type<v0::Convert>(target.get_node());
                return ov::is_type<v0::Result>(target.get_node()) ||
                       (convert && convert->get_destination_type() == element::f32);
            });
            if (shape_size(constant->get_shape()) <= fold_to_f32_max_elements && !final_form) {
                continue;
            }
            changed = read_through_convert(constant) || changed;
            kept.push_back(constant);
        } else if (const auto param = ov::as_type_ptr<v0::Parameter>(node)) {
            if (!is_narrow(param->get_element_type())) {
                continue;
            }
            // Side-channel Parameters stay narrow all the way to their consumers.
            if (!is_side_channel(param)) {
                changed = read_through_convert(param) || changed;
            }
            kept.push_back(param);
        }
    }

    // Retype the rest of the graph, including the ops whose output type is an attribute (Range, Eye,
    // RandomUniform, ...). The inputs are kept above and the outputs get their type back below.
    for (const auto& node : kept) {
        ov::enable_keep_const_precision(node);
    }
    // ConvertPrecision drops the Converts that become identities, merging their tensor names into their
    // input; for a Convert that feeds a Result that would rename the model output. Keep those Converts.
    std::vector<std::shared_ptr<Node>> output_converts;
    for (const auto& result : model->get_results()) {
        const auto convert = result->get_input_node_shared_ptr(0);
        if (ov::is_type<v0::Convert>(convert) && !ov::pass::constant_folding_is_disabled(convert)) {
            ov::pass::disable_constant_folding(convert);
            output_converts.push_back(convert);
        }
    }
    const precisions_map precisions{{element::bf16, element::f32}, {element::f16, element::f32}};
    changed = ov::pass::ConvertPrecision(precisions).run_on_model(model) || changed;
    for (const auto& node : kept) {
        ov::disable_keep_const_precision(node);
    }
    for (const auto& convert : output_converts) {
        ov::pass::enable_constant_folding(convert);
    }
    // ConvertPrecision returns false even when it retyped the graph, so `changed` only covers the read-through
    // above: f32 Parameter -> Convert(bf16) -> Relu -> Result has nothing to read through, yet its output is now f32.
    // Restore the outputs regardless.
    model->validate_nodes_and_infer_types();

    // Give every output back its original type; move the tensor names onto
    // the new Convert so the output port keeps them. Results that read the same
    // tensor (`return y, y`) share one Convert: the names move only once.
    std::map<std::pair<Output<Node>, element::Type>, Output<Node>> restored;
    const auto& results = model->get_results();
    for (size_t i = 0; i < results.size(); ++i) {
        auto source = results[i]->input_value(0);
        if (source.get_element_type() == result_types[i] || result_types[i].is_dynamic()) {
            continue;
        }
        changed = true;
        auto& convert = restored[{source, result_types[i]}];
        if (!convert.get_node()) {
            convert = std::make_shared<v0::Convert>(source, result_types[i])->output(0);
            const auto names = source.get_names();
            source.get_tensor().set_names({});
            convert.get_tensor().set_names(names);
        }
        results[i]->input(0).replace_source_output(convert);
    }

    if (changed) {
        model->validate_nodes_and_infer_types();
    }
    return changed;
}

}  // namespace ov::frontend::pytorch::pass
