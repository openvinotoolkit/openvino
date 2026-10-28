// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "canonicalize_float_precision.hpp"

#include <algorithm>
#include <map>
#include <unordered_set>
#include <utility>

#include "openvino/core/model.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/util/log.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace ov::frontend::pytorch::pass {

namespace {
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v5 = ov::op::v5;

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

// A consumer whose result jumps on a one-ulp difference of its input: integer and boolean results (Convert,
// comparisons, TopK indices, ...) and the rounding ops. ShapeOf reads only the shape.
bool is_rounding_sensitive(const Node* node) {
    if (ov::is_type_any_of<v0::ShapeOf, v3::ShapeOf>(node)) {
        return false;
    }
    if (ov::is_type_any_of<v0::Floor, v0::Ceiling, v5::Round, v0::Sign, v1::FloorMod, v1::Mod>(node)) {
        return true;
    }
    const auto outputs = node->outputs();
    return std::any_of(outputs.begin(), outputs.end(), [](const Output<const Node>& output) {
        return !output.get_element_type().is_real();
    });
}

// An explicit cast to bf16/f16 rounds: 1.999 -> bf16 2.0, i32 257 -> bf16 256. Retargeting it to f32 like the other
// narrow Converts drops that rounding; float math downstream is meant to run in f32 anyway, but a rounding-sensitive
// consumer (see above) can turn the difference into a different integer, boolean or step, which the output types
// restored at the end can't repair. A Range with a narrow output type rounds the same way: it becomes an f32 Range,
// and its sensitive consumers read it through such a cast. Gives the sensitive consumers a narrowing Convert of
// their own (shared with other consumers it would be retyped) and returns those Converts for ConvertPrecision to
// skip. A consumer other than a Convert reads it through a same-type Convert, which ConvertPrecision then widens to
// f32 like any read-through.
std::unordered_set<const Node*> isolate_rounding_casts(const std::shared_ptr<ov::Model>& model) {
    std::unordered_set<const Node*> rounding_casts;
    for (const auto& node : model->get_ordered_ops()) {
        std::shared_ptr<v0::Convert> convert;
        if (const auto range = ov::as_type_ptr<v4::Range>(node)) {
            const auto type = range->get_output_type();
            const auto targets = range->output(0).get_target_inputs();
            if (!is_narrow(type) || std::none_of(targets.begin(), targets.end(), [](const Input<Node>& target) {
                    return is_rounding_sensitive(target.get_node());
                })) {
                continue;
            }
            range->set_output_type(element::f32);
            range->validate_and_infer_types();
            convert = std::make_shared<v0::Convert>(range, type);
            convert->set_friendly_name(range->get_friendly_name() + "/rounding");
            for (auto target : targets) {
                if (is_rounding_sensitive(target.get_node())) {
                    target.replace_source_output(convert);
                }
            }
        } else {
            convert = ov::as_type_ptr<v0::Convert>(node);
            // A same-type Convert is a read-through bridge; anything else (wider float, integer, the other narrow
            // type) rounds.
            if (!convert || !is_narrow(convert->get_destination_type()) ||
                convert->get_input_element_type(0) == convert->get_destination_type()) {
                continue;
            }
        }
        std::vector<Input<Node>> sensitive;
        const auto targets = convert->output(0).get_target_inputs();
        for (const auto& target : targets) {
            if (is_rounding_sensitive(target.get_node())) {
                sensitive.push_back(target);
            }
        }
        if (sensitive.empty()) {
            continue;
        }
        auto rounding = convert;
        if (sensitive.size() != targets.size()) {
            rounding = std::make_shared<v0::Convert>(convert->input_value(0), convert->get_destination_type());
            rounding->set_friendly_name(convert->get_friendly_name() + "/rounding");
            for (auto& consumer : sensitive) {
                consumer.replace_source_output(rounding);
            }
        }
        rounding_casts.insert(rounding.get());
        std::shared_ptr<v0::Convert> read_through;
        for (auto& consumer : sensitive) {
            if (ov::is_type<v0::Convert>(consumer.get_node())) {
                continue;
            }
            if (!read_through) {
                read_through = std::make_shared<v0::Convert>(rounding, rounding->get_destination_type());
            }
            consumer.replace_source_output(read_through);
        }
    }
    return rounding_casts;
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
    const auto rounding_casts = isolate_rounding_casts(model);
    changed = changed || !rounding_casts.empty();
    // Same as ConvertPrecision's own Convert handling, except for the rounding casts, which stay narrow.
    const type_to_fuse_map keep_rounding_casts{
        {v0::Convert::get_type_info_static(),
         [&rounding_casts](const std::shared_ptr<Node>& node, const precisions_map& precisions) {
             const auto it = precisions.find(node->get_output_element_type(0));
             if (rounding_casts.count(node.get()) || it == precisions.end()) {
                 return false;
             }
             ov::as_type_ptr<v0::Convert>(node)->set_convert_element_type(it->second);
             return true;
         }}};
    const precisions_map precisions{{element::bf16, element::f32}, {element::f16, element::f32}};
    changed = ov::pass::ConvertPrecision(precisions, keep_rounding_casts).run_on_model(model) || changed;
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
