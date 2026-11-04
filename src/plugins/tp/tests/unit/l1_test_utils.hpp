// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/ops.hpp"
#include "tp_transformations/analyzer.hpp"
#include "tp_transformations/annotations.hpp"
#include "tp_transformations/rank_passes.hpp"

namespace ov::tp::tests {

/// Annotates `model` in place.
inline TPAnalysisReportPtr analyze(const std::shared_ptr<ov::Model>& model, size_t world_size, bool shard_lm_head = true) {
    auto report = std::make_shared<TPAnalysisReport>();
    TPLayerAnalyzer(TPAnalyzerConfig{world_size, shard_lm_head}, report).run_on_model(model);
    return report;
}

/// The rank model the plugin builds from an analyzed model.
inline std::shared_ptr<ov::Model> rank_model(const std::shared_ptr<const ov::Model>& analyzed, size_t rank, size_t world_size) {
    auto model = analyzed->clone();
    TPWeightShardingPass(rank, world_size).run_on_model(model);
    TPInsertCollectives(rank, world_size).run_on_model(model);
    strip_annotations(*model);
    return model;
}

inline std::vector<std::shared_ptr<ov::Model>> rank_models(const std::shared_ptr<const ov::Model>& model,
                                                           size_t world_size,
                                                           bool shard_lm_head = true) {
    auto analyzed = model->clone();
    analyze(analyzed, world_size, shard_lm_head);
    std::vector<std::shared_ptr<ov::Model>> models;
    for (size_t rank = 0; rank < world_size; ++rank)
        models.push_back(rank_model(analyzed, rank, world_size));
    return models;
}

inline size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model, const ov::DiscreteTypeInfo& type) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops())
        count += op->get_type_info().is_castable(type) ? 1 : 0;
    return count;
}

/// Projection roles by MatMul name.
inline std::map<std::string, ProjectionRole> roles(const std::shared_ptr<ov::Model>& analyzed) {
    std::map<std::string, ProjectionRole> result;
    for (const auto& node : analyzed->get_ordered_ops()) {
        if (const auto* role = get_role(*node))
            result[node->get_friendly_name()] = *role;
    }
    return result;
}

/// The data shape of every Gather over a static 2D table (an embedding), by name.
inline std::map<std::string, ov::PartialShape> gather_tables(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, ov::PartialShape> tables;
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::util::GatherBase>(node))
            continue;
        const auto& shape = node->get_input_partial_shape(0);
        if (shape.is_static() && shape.size() == 2)
            tables[node->get_friendly_name()] = shape;
    }
    return tables;
}

}  // namespace ov::tp::tests
