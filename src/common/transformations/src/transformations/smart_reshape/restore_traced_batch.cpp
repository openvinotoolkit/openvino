// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <memory>
#include <optional>

#include "itt.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;

namespace {

std::optional<int64_t> single_constant_value(const ov::Output<ov::Node>& output) {
    const auto constant = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
    if (!constant) {
        return std::nullopt;
    }
    const auto values = constant->cast_vector<int64_t>();
    if (values.size() != 1) {
        return std::nullopt;
    }
    return values.front();
}

// The Gather(ShapeOf(parameter), 0) the model already applies to other Reshape targets.
ov::Output<ov::Node> leading_dimension_expression(const std::shared_ptr<v0::Parameter>& parameter) {
    for (const auto& shape_of_input : parameter->output(0).get_target_inputs()) {
        const auto shape_of = shape_of_input.get_node();
        if (!ov::is_type<v0::ShapeOf>(shape_of) && !ov::is_type<v3::ShapeOf>(shape_of)) {
            continue;
        }
        for (const auto& gather_input : shape_of->output(0).get_target_inputs()) {
            const auto gather = ov::as_type<ov::op::util::GatherBase>(gather_input.get_node());
            if (gather && gather->get_output_partial_shape(0) == ov::PartialShape{1} &&
                single_constant_value(gather->input_value(1)) == 0) {
                return gather->output(0);
            }
        }
    }
    return {};
}

std::optional<size_t> batch_parameter_index(const std::shared_ptr<ov::Model>& model) {
    const auto& parameters = model->get_parameters();
    for (size_t index = 0; index < parameters.size(); ++index) {
        if (leading_dimension_expression(parameters[index]).get_node()) {
            return index;
        }
    }
    return std::nullopt;
}

// A target keeping the leading dimension at a constant one while a -1 absorbs the batch instead.
bool is_pinned_leading_target(const v0::Concat& target) {
    if (target.get_axis() != 0 || target.get_input_size() < 2 || single_constant_value(target.input_value(0)) != 1) {
        return false;
    }
    for (size_t index = 1; index < target.get_input_size(); ++index) {
        if (single_constant_value(target.input_value(index)) == -1) {
            return true;
        }
    }
    return false;
}

// Keeps the symbolic analysis off models that hold nothing this transformation could rewrite.
bool has_pinned_leading_target(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<v1::Reshape>(node)) {
            continue;
        }
        const auto target = ov::as_type_ptr<v0::Concat>(node->input_value(1).get_node_shared_ptr());
        if (target && is_pinned_leading_target(*target)) {
            return true;
        }
    }
    return false;
}

// Makes the leading dimension of one input dynamic so a batch-dependent tensor is recognizable no matter which shape
// the model was stored with. Other inputs keep their shapes, so a batch-independent source stays recognizable too.
ov::PartialShape open_leading_dimension(const std::shared_ptr<ov::Model>& model, size_t input_index) {
    const auto& parameter = model->get_parameters()[input_index];
    const auto original_shape = parameter->get_partial_shape();
    auto opened_shape = original_shape;
    if (opened_shape.rank().is_dynamic() || opened_shape.size() == 0) {
        return original_shape;
    }
    opened_shape[0] = ov::Dimension::dynamic();
    parameter->set_partial_shape(opened_shape);
    model->validate_nodes_and_infer_types();
    return original_shape;
}

void restore_leading_dimension(const std::shared_ptr<ov::Model>& model,
                               size_t input_index,
                               const ov::PartialShape& original_shape) {
    model->get_parameters()[input_index]->set_partial_shape(original_shape);
    model->validate_nodes_and_infer_types();
}

// A Reshape taking the batch from a shape expression while its data keeps a static leading dimension proves that an
// earlier Reshape pinned the batch.
bool batch_is_rematerialized(const ov::Model& model, size_t input_index) {
    const auto probe = model.clone();
    const auto& parameter = probe->get_parameters()[input_index];
    auto seeded_shape = parameter->get_partial_shape();
    if (seeded_shape.rank().is_dynamic() || seeded_shape.size() == 0) {
        return false;
    }
    seeded_shape[0] = ov::Dimension::dynamic();
    seeded_shape[0].set_symbol(std::make_shared<ov::Symbol>());
    parameter->set_partial_shape(seeded_shape);
    probe->validate_nodes_and_infer_types();

    ov::pass::SymbolicPropagation().run_on_model(probe);

    const auto& propagated_shape = parameter->get_partial_shape();
    if (!propagated_shape[0].has_symbol()) {
        return false;
    }
    const auto batch = propagated_shape[0].get_symbol();

    for (const auto& node : probe->get_ordered_ops()) {
        if (!ov::is_type<v1::Reshape>(node)) {
            continue;
        }
        const auto& data_shape = node->get_input_partial_shape(0);
        const auto& result_shape = node->get_output_partial_shape(0);
        if (data_shape.rank().is_dynamic() || data_shape.size() == 0 || result_shape.rank().is_dynamic() ||
            result_shape.size() == 0) {
            continue;
        }
        if (data_shape[0].is_static() && result_shape[0].has_symbol() &&
            ov::symbol::are_equal(result_shape[0].get_symbol(), batch)) {
            return true;
        }
    }
    return false;
}

}  // namespace

namespace ov::pass {

// Rewrites the pinned leading element of a Reshape target. Concat is variadic, so its arity stays out of the pattern
// and the target layout is checked in the callback.
class RestorePinnedLeadingDimension : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RestorePinnedLeadingDimension");

    explicit RestorePinnedLeadingDimension(const std::shared_ptr<v0::Parameter>& batch_parameter) {
        MATCHER_SCOPE(RestorePinnedLeadingDimension);
        const auto leading_dimension = leading_dimension_expression(batch_parameter);
        const auto data = ov::pass::pattern::any_input();
        const auto target = ov::pass::pattern::wrap_type<v0::Concat>();
        const auto reshape = ov::pass::pattern::wrap_type<v1::Reshape>({data, target});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& matcher) {
            const auto& pattern_map = matcher.get_pattern_value_map();
            const auto& data_shape = pattern_map.at(data).get_partial_shape();
            if (data_shape.rank().is_static() && (data_shape.size() == 0 || data_shape[0].is_static())) {
                return false;
            }

            const auto target_shape = ov::as_type_ptr<v0::Concat>(pattern_map.at(target).get_node_shared_ptr());
            if (!is_pinned_leading_target(*target_shape) ||
                target_shape->get_element_type() != leading_dimension.get_element_type()) {
                return false;
            }

            target_shape->input(0).replace_source_output(leading_dimension);
            return true;
        };

        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(reshape, matcher_name), callback);
    }
};

}  // namespace ov::pass

bool ov::pass::RestoreTracedBatch::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RestoreTracedBatch);

    const auto input_index = batch_parameter_index(model);
    if (!input_index || !has_pinned_leading_target(model) || !batch_is_rematerialized(*model, *input_index)) {
        return false;
    }

    const auto original_shape = open_leading_dimension(model, *input_index);

    ov::pass::Manager manager("RestoreTracedBatch:rewrite");
    manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<RestorePinnedLeadingDimension>(
        model->get_parameters()[*input_index]);
    const bool model_changed = manager.run_passes(model);

    restore_leading_dimension(model, *input_index, original_shape);
    return model_changed;
}
