// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/rematerialized_batch.hpp"

#include <memory>

#include "openvino/core/symbol.hpp"
#include "openvino/op/reshape.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

std::optional<int64_t> ov::find_rematerialized_batch(const ov::Model& model, size_t input_index) {
    if (input_index >= model.get_parameters().size()) {
        return std::nullopt;
    }

    const auto probe = model.clone();
    const auto& parameter = probe->get_parameters()[input_index];
    auto seeded_shape = parameter->get_partial_shape();
    if (seeded_shape.rank().is_dynamic() || seeded_shape.size() == 0) {
        return std::nullopt;
    }
    seeded_shape[0] = ov::Dimension::dynamic();
    seeded_shape[0].set_symbol(std::make_shared<ov::Symbol>());
    parameter->set_partial_shape(seeded_shape);
    probe->validate_nodes_and_infer_types();

    ov::pass::SymbolicPropagation().run_on_model(probe);

    const auto& propagated_shape = parameter->get_partial_shape();
    if (propagated_shape.rank().is_dynamic() || propagated_shape.size() == 0 || !propagated_shape[0].has_symbol()) {
        return std::nullopt;
    }
    const auto batch = propagated_shape[0].get_symbol();

    for (const auto& node : probe->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v1::Reshape>(node)) {
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
            return data_shape[0].get_length();
        }
    }
    return std::nullopt;
}
