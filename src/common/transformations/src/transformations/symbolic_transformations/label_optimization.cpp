// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/label_optimization.hpp"

#include <openvino/core/dimension_tracker.hpp>
#include <transformations/symbolic_transformations/utils.hpp>

#include "itt.hpp"
#include "openvino/op/util/symbolic_info.hpp"

bool ov::pass::ApplyTableOfEquivalence::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(ApplyTableOfEquivalence);

    auto te = ov::table_of_equivalence(m);
    if (te == nullptr)
        return false;
    auto equivalence_table = te->get_equivalence_table();

    for (const auto& op : m->get_ordered_ops()) {
        for (auto& output : op->outputs()) {
            auto shape = output.get_partial_shape();
            for (auto& d : shape) {
                if (d.is_static())
                    continue;
                auto label = ov::DimensionTracker::get_label(d);
                if (label != ov::no_label && equivalence_table.count(label) &&
                    *equivalence_table[label]->begin() != ov::no_label) {
                    label = std::min(label, *equivalence_table[label]->begin());
                    ov::DimensionTracker::set_label(d, label);
                }
            }
            op->set_output_type(output.get_index(), output.get_element_type(), shape);
            auto value_labels = output.get_tensor().get_value_label();
            if (!value_labels.empty() &&
                !std::all_of(value_labels.begin(), value_labels.end(), [](const ov::label_t& i) {
                    return i == ov::no_label;
                })) {
                for (auto& label : value_labels) {
                    if (equivalence_table.count(label) && *equivalence_table[label]->begin() != ov::no_label)
                        label = std::min(label, *equivalence_table[label]->begin());
                }
                output.get_tensor().set_value_label(value_labels);
            }
        }
    }
    return false;
}

bool ov::pass::OptimizeLabelsUsedAsValues::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(OptimizeLabelsUsedAsValues);
    using LTS_map = std::unordered_map<ov::label_t, ov::Output<ov::Node>>;
    LTS_map label_shape_source;
    LTS_map label_value_source;
    for (const auto& op : m->get_ordered_ops()) {
        for (auto& output : op->outputs()) {
            optimize_value_usage(output, label_shape_source, label_value_source);
            save_shape_sources(output, label_shape_source);
        }
    }
    return true;
}
