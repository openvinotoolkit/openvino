// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "canonicalize_float_precision.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

namespace {
namespace v0 = ov::op::v0;

// Narrow Constants up to this many elements (scalars, eps, rotary tables) are
// folded to f32 instead of kept behind a decompression Convert.
constexpr size_t fold_to_f32_max_elements = 1024;

bool is_narrow(const element::Type& type) {
    return type == element::bf16 || type == element::f16;
}

bool is_side_channel(const std::shared_ptr<v0::Parameter>& param) {
    return param->get_friendly_name().rfind("__pa__", 0) == 0;
}
}  // namespace

bool CanonicalizeFloatPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<element::Type> result_types;
    for (const auto& result : model->get_results()) {
        result_types.push_back(result->get_input_element_type(0));
    }

    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto convert = ov::as_type_ptr<v0::Convert>(node)) {
            if (is_narrow(convert->get_destination_type())) {
                convert->set_destination_type(element::f32);
                changed = true;
            }
        } else if (const auto constant = ov::as_type_ptr<v0::Constant>(node)) {
            if (!is_narrow(constant->get_element_type())) {
                continue;
            }
            std::vector<Input<Node>> consumers;
            for (const auto& input : constant->output(0).get_target_inputs()) {
                const auto consumer = ov::as_type<v0::Convert>(input.get_node());
                if (!consumer || consumer->get_destination_type() != element::f32) {
                    consumers.push_back(input);
                }
            }
            if (consumers.empty()) {
                continue;  // already in decompression form
            }
            std::shared_ptr<Node> replacement;
            if (shape_size(constant->get_shape()) <= fold_to_f32_max_elements) {
                replacement =
                    std::make_shared<v0::Constant>(element::f32, constant->get_shape(), constant->cast_vector<float>());
            } else {
                replacement = std::make_shared<v0::Convert>(constant, element::f32);
            }
            for (auto& consumer : consumers) {
                consumer.replace_source_output(replacement);
            }
            changed = true;
        } else if (const auto param = ov::as_type_ptr<v0::Parameter>(node)) {
            if (!is_narrow(param->get_element_type()) || is_side_channel(param)) {
                continue;
            }
            const auto consumers = param->output(0).get_target_inputs();
            const auto convert = std::make_shared<v0::Convert>(param, element::f32);
            for (auto consumer : consumers) {
                consumer.replace_source_output(convert);
            }
            changed = true;
        }
    }
    if (!changed) {
        return false;
    }
    model->validate_nodes_and_infer_types();

    // Give every output back its original type; move the tensor names onto
    // the new Convert so the output port keeps them.
    const auto& results = model->get_results();
    for (size_t i = 0; i < results.size(); ++i) {
        auto source = results[i]->input_value(0);
        if (source.get_element_type() == result_types[i] || result_types[i].is_dynamic()) {
            continue;
        }
        const auto convert = std::make_shared<v0::Convert>(source, result_types[i]);
        const auto names = source.get_names();
        source.get_tensor().set_names({});
        convert->output(0).get_tensor().set_names(names);
        results[i]->input(0).replace_source_output(convert);
    }

    // Converts that used to bridge a narrow island and an f32 one are now
    // identities; drop them so they don't sit between ops the fusions match.
    for (const auto& node : model->get_ordered_ops()) {
        const auto convert = ov::as_type_ptr<v0::Convert>(node);
        if (!convert) {
            continue;
        }
        const auto source = convert->input_value(0);
        if (source.get_element_type() != convert->get_destination_type()) {
            continue;
        }
        const auto targets = convert->output(0).get_target_inputs();
        const bool feeds_result = std::any_of(targets.begin(), targets.end(), [](const Input<Node>& target) {
            return ov::is_type<v0::Result>(target.get_node());
        });
        if (feeds_result) {
            continue;
        }
        for (auto target : targets) {
            target.replace_source_output(source);
        }
    }

    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
