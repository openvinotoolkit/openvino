// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "materialize_weights.hpp"

#include <map>

#include "helper_ops/gguf_constant.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "quant/deferred_weights.hpp"

namespace ov::frontend::gguf::pass {
namespace {

void fuse_projections(const std::shared_ptr<ov::Model>& model) {
    using MatMul = ov::op::v0::MatMul;
    std::map<std::pair<ov::Node*, size_t>, std::vector<std::shared_ptr<MatMul>>> groups;
    for (const auto& node : model->get_ordered_ops()) {
        const auto matmul = ov::as_type_ptr<MatMul>(node);
        if (!matmul || matmul->get_transpose_a() || !matmul->get_transpose_b()) {
            continue;
        }
        const auto weight = ov::as_type_ptr<GGUFConstant>(matmul->get_input_node_shared_ptr(1));
        if (weight && weight->tensors().weight && weight->is_u2() && weight->get_output_shape(0).size() == 2) {
            const auto input = matmul->input_value(0);
            groups[{input.get_node(), input.get_index()}].push_back(matmul);
        }
    }
    for (const auto& [input, candidates] : groups) {
        std::vector<bool> used(candidates.size());
        for (size_t first = 0; first < candidates.size(); ++first) {
            if (used[first]) {
                continue;
            }
            const auto first_weight = ov::as_type_ptr<GGUFConstant>(candidates[first]->get_input_node_shared_ptr(1));
            auto tensors = first_weight->tensors();
            std::vector<std::shared_ptr<MatMul>> projections{candidates[first]};
            ov::OutputVector weights{first_weight};
            std::vector<int64_t> lengths{static_cast<int64_t>(first_weight->get_output_shape(0)[0])};
            for (size_t i = first + 1; i < candidates.size(); ++i) {
                if (used[i]) {
                    continue;
                }
                const auto weight = ov::as_type_ptr<GGUFConstant>(candidates[i]->get_input_node_shared_ptr(1));
                if (!first_weight->can_concat(*weight)) {
                    continue;
                }
                auto combined = concat_deferred_rows(tensors, weight->tensors());
                if (!combined) {
                    continue;
                }
                tensors = *combined;
                projections.push_back(candidates[i]);
                weights.push_back(weight);
                lengths.push_back(static_cast<int64_t>(weight->get_output_shape(0)[0]));
                used[i] = true;
            }
            if (projections.size() == 1) {
                continue;
            }
            auto concat = std::make_shared<ov::op::v0::Concat>(weights, 0);
            auto merged = std::make_shared<MatMul>(projections.front()->input_value(0), concat, false, true);
            merged->set_friendly_name(projections.front()->get_friendly_name() + "/merged");
            auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {-1});
            auto sizes = ov::op::v0::Constant::create(ov::element::i64, {lengths.size()}, lengths);
            auto split = std::make_shared<ov::op::v1::VariadicSplit>(merged, axis, sizes);
            ov::NodeVector sources(projections.begin(), projections.end());
            ov::copy_runtime_info(sources, {concat, merged, split});
            for (size_t i = 0; i < projections.size(); ++i) {
                ov::replace_node(projections[i], ov::OutputVector{split->output(i)});
            }
        }
    }
}

void combine_weights(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node);
        if (!concat || concat->get_axis() != 0 || concat->get_input_size() < 2) {
            continue;
        }
        const auto first = ov::as_type_ptr<GGUFConstant>(concat->get_input_node_shared_ptr(0));
        if (!first || !first->tensors().weight) {
            continue;
        }
        auto tensors = first->tensors();
        bool compatible = true;
        for (size_t i = 1; i < concat->get_input_size(); ++i) {
            const auto weight = ov::as_type_ptr<GGUFConstant>(concat->get_input_node_shared_ptr(i));
            if (!weight || !first->can_concat(*weight)) {
                compatible = false;
                break;
            }
            auto combined = concat_deferred_rows(tensors, weight->tensors());
            if (!combined) {
                compatible = false;
                break;
            }
            tensors = *combined;
        }
        if (compatible) {
            auto merged = std::make_shared<GGUFConstant>(tensors, first->qtype(), first->source_name());
            merged->set_friendly_name(concat->get_friendly_name());
            ov::copy_runtime_info(concat, merged);
            ov::replace_node(concat, merged);
        }
    }
}

}  // namespace

bool MaterializeGGUFWeights::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto subgraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (const auto& body : subgraph->get_functions()) {
                changed |= run_on_model(body);
            }
        }
    }
    fuse_projections(model);
    combine_weights(model);
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto weight = ov::as_type_ptr<GGUFConstant>(node)) {
            auto materialized = weight->materialize();
            OPENVINO_ASSERT(materialized->get_output_partial_shape(0) == weight->get_output_partial_shape(0) &&
                                materialized->get_output_element_type(0) == weight->get_output_element_type(0),
                            "Materialized GGUF weight has unexpected shape or element type: ",
                            weight->source_name());
            materialized->set_friendly_name(weight->get_friendly_name());
            ov::copy_runtime_info(weight, materialized);
            ov::replace_node(weight, materialized);
            changed = true;
        }
    }
    return changed;
}

}  // namespace ov::frontend::gguf::pass
