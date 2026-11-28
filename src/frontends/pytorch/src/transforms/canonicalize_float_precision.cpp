// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "canonicalize_float_precision.hpp"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/squeeze_base.hpp"
#include "openvino/op/variadic_split.hpp"
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
namespace v8 = ov::op::v8;

// Narrow Constants up to this many elements (scalars, eps, rotary tables) are folded to f32 instead.
constexpr size_t fold_to_f32_max_elements = 1024;

bool is_narrow(const element::Type& type) {
    return type == element::bf16 || type == element::f16;
}

bool is_side_channel(const std::shared_ptr<v0::Parameter>& param) {
    return param->get_friendly_name().rfind("__pa__", 0) == 0;
}

// A consumer whose result jumps on a one-ulp input difference: integer/boolean results and rounding ops.
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

// Passes its float input on unchanged (same or wider float cast, data movement), rounded value included.
bool is_value_preserving(const Node* node) {
    if (const auto convert = ov::as_type<const v0::Convert>(node)) {
        const auto& src = convert->get_input_element_type(0);
        const auto& dst = convert->get_destination_type();
        return dst.is_real() && (dst == src || dst.bitwidth() > src.bitwidth());
    }
    return ov::is_type_any_of<v1::Reshape,
                              ov::op::util::SqueezeBase,
                              v0::Unsqueeze,
                              v1::Transpose,
                              ov::op::util::GatherBase,
                              v1::StridedSlice,
                              v8::Slice,
                              v0::Concat,
                              ov::op::util::BroadcastBase,
                              v0::Tile,
                              v1::Split,
                              v1::VariadicSplit>(node);
}

// True if the node is rounding-sensitive or reaches a rounding-sensitive consumer through
// value-preserving ops only (x.to(bf16).float().to(int32)).
bool reaches_rounding_sensitive(const Node* node, std::unordered_map<const Node*, bool>& memo) {
    if (const auto it = memo.find(node); it != memo.end()) {
        return it->second;
    }
    bool reaches = is_rounding_sensitive(node);
    if (!reaches && is_value_preserving(node)) {
        for (const auto& output : node->outputs()) {
            const auto targets = output.get_target_inputs();
            reaches = std::any_of(targets.begin(), targets.end(), [&memo](const Input<Node>& target) {
                return reaches_rounding_sensitive(target.get_node(), memo);
            });
            if (reaches) {
                break;
            }
        }
    }
    memo[node] = reaches;
    return reaches;
}

// An explicit cast to bf16/f16 (or a narrow Range) rounds, e.g. 1.999 -> 2.0, and retyping it to f32
// would let a rounding-sensitive consumer produce a different integer, boolean or step. Gives those
// consumers a narrowing Convert of their own and returns them for ConvertPrecision to skip.
std::unordered_set<const Node*> isolate_rounding_casts(const std::shared_ptr<ov::Model>& model) {
    std::unordered_set<const Node*> rounding_casts;
    std::unordered_map<const Node*, bool> memo;
    const auto sensitive_target = [&memo](const Input<Node>& target) {
        return reaches_rounding_sensitive(target.get_node(), memo);
    };
    for (const auto& node : model->get_ordered_ops()) {
        std::shared_ptr<v0::Convert> convert;
        if (const auto range = ov::as_type_ptr<v4::Range>(node)) {
            const auto type = range->get_output_type();
            const auto targets = range->output(0).get_target_inputs();
            if (!is_narrow(type) || std::none_of(targets.begin(), targets.end(), sensitive_target)) {
                continue;
            }
            range->set_output_type(element::f32);
            range->validate_and_infer_types();
            convert = std::make_shared<v0::Convert>(range, type);
            convert->set_friendly_name(range->get_friendly_name() + "/rounding");
            for (auto target : targets) {
                if (sensitive_target(target)) {
                    target.replace_source_output(convert);
                }
            }
        } else {
            convert = ov::as_type_ptr<v0::Convert>(node);
            // A same-type Convert is a read-through bridge; any other cast rounds.
            if (!convert || !is_narrow(convert->get_destination_type()) ||
                convert->get_input_element_type(0) == convert->get_destination_type()) {
                continue;
            }
        }
        std::vector<Input<Node>> sensitive;
        const auto targets = convert->output(0).get_target_inputs();
        for (const auto& target : targets) {
            if (sensitive_target(target)) {
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
    // Bodies keep their converted types, so retyping the outer graph would fail validation.
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

    // Weights and inputs stay narrow; consumers read them through a same-type Convert, which
    // ConvertPrecision then retargets to f32.
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
            // Small Constants are folded to f32 by ConvertPrecision; skip one already read only by a Convert(f32)
            // or a Result.
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

    // Retype the rest, including ops with an output-type attribute (Range, Eye, RandomUniform).
    for (const auto& node : kept) {
        ov::enable_keep_const_precision(node);
    }
    // ConvertPrecision drops identity Converts and merges their tensor names into the input, which would
    // rename a model output; keep the Converts feeding a Result.
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
    // ConvertPrecision returns false even when it retyped the graph, so restore the outputs regardless.
    model->validate_nodes_and_infer_types();

    // Give every output back its type; Results reading one tensor share a Convert, names move once.
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
