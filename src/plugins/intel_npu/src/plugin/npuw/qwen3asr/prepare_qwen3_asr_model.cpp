// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prepare_qwen3_asr_model.hpp"

#include <unordered_set>

#include "../logging.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/validate.hpp"

namespace {

using Parameter = ov::op::v0::Parameter;
using Constant = ov::op::v0::Constant;
using Reshape = ov::op::v1::Reshape;
using LessEqual = ov::op::v1::LessEqual;

bool constant_is(const std::shared_ptr<ov::Node>& node, int64_t value) {
    const auto constant = ov::as_type_ptr<Constant>(node);
    if (!constant || !constant->get_element_type().is_integral_number())
        return false;
    const auto values = constant->cast_vector<int64_t>();
    return values.size() == 1 && values[0] == value;
}

bool is_causal_mask(const std::shared_ptr<ov::Node>& node) {
    const auto less_equal = ov::as_type_ptr<LessEqual>(node);
    if (!less_equal)
        return false;
    auto key = less_equal->get_input_node_shared_ptr(0);
    if (ov::is_type<ov::op::v0::Convert>(key))
        key = key->get_input_node_shared_ptr(0);
    for (size_t i = 0; i < 3; ++i) {
        const auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(key);
        if (!unsqueeze)
            return false;
        key = unsqueeze->get_input_node_shared_ptr(0);
    }
    const auto range = ov::as_type_ptr<ov::op::v4::Range>(key);
    return range && constant_is(range->get_input_node_shared_ptr(0), 0);
}

bool is_position_reshape(const std::shared_ptr<ov::Node>& node) {
    const auto reshape = ov::as_type_ptr<Reshape>(node);
    if (!reshape)
        return false;
    const auto range = ov::as_type_ptr<ov::op::v4::Range>(reshape->get_input_node_shared_ptr(0));
    if (!range)
        return false;
    const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(range->get_input_node_shared_ptr(0));
    return gather && ov::is_type<ov::op::util::ShapeOfBase>(gather->get_input_node_shared_ptr(0));
}

std::shared_ptr<Parameter> find_input(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    std::shared_ptr<Parameter> result;
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->output(0).get_names().count(name)) {
            OPENVINO_ASSERT(!result, "Qwen3-ASR model has multiple '", name, "' inputs.");
            result = parameter;
        }
    }
    return result;
}

std::shared_ptr<Parameter> make_input(const std::string& name, const ov::PartialShape& shape) {
    auto parameter = std::make_shared<Parameter>(ov::element::i64, shape);
    parameter->output(0).set_names({name});
    parameter->set_friendly_name(name);
    return parameter;
}

// Verify the actual graph, not just input names: this request writes 0 for an
// attended token, whereas standard LLM attention masks use 1 for an attended token.
bool has_prepared_mask(const std::shared_ptr<LessEqual>& causal,
                       const std::shared_ptr<Parameter>& mask,
                       std::unordered_set<const ov::Node*>& mask_consumers) {
    const auto targets = causal->output(0).get_target_inputs();
    if (targets.size() != 1)
        return false;
    const auto target = *targets.begin();
    const auto combined = ov::as_type_ptr<ov::op::v1::LogicalAnd>(target.get_node()->shared_from_this());
    if (!combined || target.get_index() != 0)
        return false;
    const auto equal = ov::as_type_ptr<ov::op::v1::Equal>(combined->get_input_node_shared_ptr(1));
    if (!equal || !constant_is(equal->get_input_node_shared_ptr(1), 0))
        return false;
    const auto u2 = ov::as_type_ptr<ov::op::v0::Unsqueeze>(equal->get_input_node_shared_ptr(0));
    if (!u2 || !constant_is(u2->get_input_node_shared_ptr(1), 2))
        return false;
    const auto u1 = ov::as_type_ptr<ov::op::v0::Unsqueeze>(u2->get_input_node_shared_ptr(0));
    if (!u1 || !constant_is(u1->get_input_node_shared_ptr(1), 1) || u1->get_input_node_shared_ptr(0) != mask)
        return false;
    mask_consumers.insert(u1.get());
    return true;
}

void validate_prepared(const std::shared_ptr<Parameter>& mask,
                       const std::shared_ptr<Parameter>& position,
                       const std::vector<std::shared_ptr<LessEqual>>& causal_masks,
                       const std::vector<std::shared_ptr<Reshape>>& position_reshapes,
                       bool prefill) {
    OPENVINO_ASSERT(mask && position && mask->get_element_type() == ov::element::i64 &&
                        mask->get_partial_shape().compatible(ov::PartialShape{1, -1}) &&
                        position->get_element_type() == ov::element::i64 &&
                        position->get_partial_shape().compatible(prefill ? ov::PartialShape{-1} : ov::PartialShape{1}),
                    "Qwen3-ASR existing attention_mask/position_ids have an unsupported type or shape.");
    OPENVINO_ASSERT(!causal_masks.empty() && position_reshapes.empty(),
                    "Qwen3-ASR existing attention_mask/position_ids do not match the prepared graph.");
    std::unordered_set<const ov::Node*> mask_consumers;
    for (const auto& causal : causal_masks) {
        OPENVINO_ASSERT(has_prepared_mask(causal, mask, mask_consumers),
                        "Qwen3-ASR existing attention_mask is not connected with 0=attend semantics.");
    }
    const auto mask_targets = mask->output(0).get_target_inputs();
    OPENVINO_ASSERT(!mask_targets.empty(), "Qwen3-ASR attention_mask is not connected.");
    for (const auto& target : mask_targets) {
        OPENVINO_ASSERT(target.get_index() == 0 && mask_consumers.count(target.get_node()),
                        "Qwen3-ASR attention_mask has an unsupported consumer.");
    }
    const auto position_targets = position->output(0).get_target_inputs();
    OPENVINO_ASSERT(!position_targets.empty(), "Qwen3-ASR position_ids is not connected.");
    for (const auto& target : position_targets) {
        const auto reshape = ov::as_type_ptr<Reshape>(target.get_node()->shared_from_this());
        OPENVINO_ASSERT(reshape && target.get_index() == 0, "Qwen3-ASR position_ids has an unsupported consumer.");
        const auto shape = ov::as_type_ptr<Constant>(reshape->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(shape && shape->get_element_type().is_integral_number(),
                        "Qwen3-ASR position reshape must have a constant integer shape.");
        const auto values = shape->cast_vector<int64_t>();
        OPENVINO_ASSERT(!values.empty() && (!prefill || values.back() == -1),
                        "Qwen3-ASR position reshape has an unsupported shape.");
    }
}

bool prepare_decoder(const std::shared_ptr<ov::Model>& model, bool prefill) {
    std::vector<std::shared_ptr<LessEqual>> causal_masks;
    std::vector<std::shared_ptr<Reshape>> position_reshapes;
    for (const auto& node : model->get_ordered_ops()) {
        if (is_causal_mask(node))
            causal_masks.push_back(ov::as_type_ptr<LessEqual>(node));
        if (is_position_reshape(node))
            position_reshapes.push_back(ov::as_type_ptr<Reshape>(node));
    }

    const auto existing_mask = find_input(model, "attention_mask");
    const auto existing_position = find_input(model, "position_ids");
    if (existing_mask || existing_position) {
        validate_prepared(existing_mask, existing_position, causal_masks, position_reshapes, prefill);
        return false;
    }

    OPENVINO_ASSERT(!causal_masks.empty() && !position_reshapes.empty(),
                    "Qwen3-ASR model preparation requires supported attention and position patterns.");
    for (const auto& reshape : position_reshapes) {
        const auto shape = ov::as_type_ptr<Constant>(reshape->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(
            shape && shape->get_element_type().is_integral_number() && !shape->cast_vector<int64_t>().empty(),
            "Qwen3-ASR position reshape must have a nonempty constant integer shape.");
    }

    const auto mask = make_input("attention_mask", ov::PartialShape{1, -1});
    const auto position = make_input("position_ids", prefill ? ov::PartialShape{-1} : ov::PartialShape{1});
    ov::copy_runtime_info(causal_masks.front(), mask);
    ov::copy_runtime_info(position_reshapes.front(), position);
    model->add_parameters({mask, position});

    for (const auto& causal : causal_masks) {
        const auto targets = causal->output(0).get_target_inputs();
        const auto zero = Constant::create(ov::element::i64, ov::Shape{}, {0});
        const auto one = Constant::create(ov::element::i64, ov::Shape{}, {1});
        const auto two = Constant::create(ov::element::i64, ov::Shape{}, {2});
        const auto u1 = std::make_shared<ov::op::v0::Unsqueeze>(mask, one);
        const auto u2 = std::make_shared<ov::op::v0::Unsqueeze>(u1, two);
        const auto equal = std::make_shared<ov::op::v1::Equal>(u2, zero);
        const auto combined = std::make_shared<ov::op::v1::LogicalAnd>(causal, equal);
        ov::copy_runtime_info(causal, {zero, one, two, u1, u2, equal, combined});
        ov::copy_output_runtime_info({causal->output(0)}, {combined->output(0)});
        combined->set_friendly_name(causal->get_friendly_name());
        causal->set_friendly_name(causal->get_friendly_name() + "/causal");
        combined->output(0).set_names(causal->output(0).get_names());
        causal->output(0).set_names({});
        for (auto target : targets)
            target.replace_source_output(combined->output(0));
    }

    for (const auto& reshape : position_reshapes) {
        if (prefill) {
            const auto shape = ov::as_type_ptr<Constant>(reshape->get_input_node_shared_ptr(1));
            auto values = shape->cast_vector<int64_t>();
            if (values.back() != -1) {
                values.back() = -1;
                const auto replacement = Constant::create(shape->get_element_type(), shape->get_shape(), values);
                ov::copy_runtime_info(shape, replacement);
                ov::copy_output_runtime_info({shape->output(0)}, {replacement->output(0)});
                if (shape->output(0).get_target_inputs().size() == 1) {
                    replacement->set_friendly_name(shape->get_friendly_name());
                    replacement->output(0).set_names(shape->output(0).get_names());
                } else {
                    replacement->set_friendly_name(shape->get_friendly_name() + "/qwen3asr");
                }
                reshape->input(1).replace_source_output(replacement);
            }
        }
        reshape->input(0).replace_source_output(position);
    }

    ov::pass::Validate().run_on_model(model);
    return true;
}

}  // namespace

bool ov::npuw::PrepareQwen3ASRModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PrepareQwen3ASRModel);
    LOG_DEBUG("[Qwen3-ASR] Removing residual state tensors.");
    bool changed = false;
    for (const auto& op : model->get_ops()) {
        const auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(op);
        if (!read_value)
            continue;
        const auto var_id = read_value->get_variable_id();
        std::shared_ptr<Parameter> replacement;
        if (read_value->get_input_size() > 0)
            replacement = ov::as_type_ptr<Parameter>(read_value->get_input_node_shared_ptr(0));
        if (!replacement) {
            replacement = std::make_shared<Parameter>(read_value->get_output_element_type(0),
                                                      read_value->get_output_partial_shape(0));
            replacement->output(0).set_names({var_id});
            replacement->set_friendly_name(read_value->get_friendly_name());
            model->add_parameters({replacement});
        }
        ov::copy_runtime_info(read_value, replacement);
        ov::copy_output_runtime_info({read_value->output(0)}, {replacement->output(0)});
        auto names = replacement->output(0).get_names();
        const auto& read_names = read_value->output(0).get_names();
        names.insert(read_names.begin(), read_names.end());
        replacement->output(0).set_names(names);
        ov::replace_node(read_value, replacement);
        changed = true;
    }
    const auto sinks = model->get_sinks();
    for (const auto& sink : sinks) {
        if (ov::is_type<ov::op::util::AssignBase>(sink)) {
            model->remove_sink(sink);
            changed = true;
        }
    }
    const auto variables = model->get_variables();
    for (const auto& var : variables) {
        model->remove_variable(var);
        changed = true;
    }
    return changed;
}

bool ov::npuw::PrepareQwen3ASRKVCacheModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PrepareQwen3ASRKVCacheModel);
    return prepare_decoder(model, false);
}

bool ov::npuw::PrepareQwen3ASRPrefillModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PrepareQwen3ASRPrefillModel);
    return prepare_decoder(model, true);
}
