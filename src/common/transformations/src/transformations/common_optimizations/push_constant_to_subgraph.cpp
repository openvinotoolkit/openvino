// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "validation_util.hpp"

using MultiSubGraphOp = ov::op::util::MultiSubGraphOp;

namespace {
struct ConstantToPushDesc {
    ConstantToPushDesc(const uint64_t body_idx,
                       const uint64_t body_param_idx,
                       const uint64_t op_in_idx,
                       const std::shared_ptr<ov::op::v0::Constant>& constant,
                       const int64_t body_param_consumers,
                       const bool used_in_many_bodies)
        : m_body_idx{body_idx},
          m_body_param_idx{body_param_idx},
          m_op_in_idx{op_in_idx},
          m_constant{constant},
          m_body_param_consumers{body_param_consumers},
          m_used_in_many_bodies{used_in_many_bodies} {}

    const size_t m_body_idx = -1;
    uint64_t m_body_param_idx = -1;
    const uint64_t m_op_in_idx = -1;
    const std::shared_ptr<ov::op::v0::Constant> m_constant;
    const int64_t m_body_param_consumers = -1;
    bool m_used_in_many_bodies = false;
};
}  // namespace

static void replace_body_parameter(const std::shared_ptr<ov::Model>& body,
                                   const std::shared_ptr<ov::op::v0::Parameter>& body_param,
                                   const ConstantToPushDesc& const_to_push,
                                   MultiSubGraphOp::MultiSubgraphInputDescriptionVector& descriptions,
                                   std::unordered_multimap<size_t, ConstantToPushDesc>& cache) {
    body_param->output(0).replace(const_to_push.m_constant);
    body->remove_parameter(body_param);
    // update all input descriptions to reflect that body parameter was removed
    for (auto& desc : descriptions) {
        if (desc->m_body_parameter_index > const_to_push.m_body_param_idx) {
            desc->m_body_parameter_index--;
        }
    }
    for (auto& in : cache) {
        if (in.second.m_body_idx == const_to_push.m_body_idx &&
            in.second.m_body_param_idx > const_to_push.m_body_param_idx) {
            in.second.m_body_param_idx--;
        }
    }
}

static void update_multi_sub_graph_op_inputs(const std::shared_ptr<MultiSubGraphOp>& multi_sub_graph_op,
                                             const std::vector<bool>& remove_inputs_mask) {
    int num_subgraphs = static_cast<int>(multi_sub_graph_op->get_internal_subgraphs_size());
    auto inputs = multi_sub_graph_op->input_values();
    for (size_t i = multi_sub_graph_op->get_input_size(); i > 0; i--) {
        const auto input_index = i - 1;
        if (remove_inputs_mask[input_index]) {
            // remove MultiSubGraphOp's input if it was marked to be removed
            // (meaning it was constfolded and pushed to inner subgraph)
            inputs.erase(inputs.begin() + input_index);

            // update input descriptions to reflect that the input was removed
            for (int body_idx = 0; body_idx < num_subgraphs; body_idx++) {
                auto& descriptions = multi_sub_graph_op->get_input_descriptions(body_idx);
                for (auto& desc : descriptions) {
                    if (desc->m_input_index > input_index) {
                        desc->m_input_index--;
                    }
                }
            }
        }
    }
    multi_sub_graph_op->set_arguments(inputs);
}

bool ov::pass::PushConstantToSubgraph::run_on_model(const std::shared_ptr<Model>& model) {
    RUN_ON_FUNCTION_SCOPE(PushConstantToSubgraph);

    bool result = false;
    for (const auto& op : model->get_ordered_ops()) {
        const auto multi_sub_graph_op = as_type_ptr<op::util::MultiSubGraphOp>(op);
        if (!multi_sub_graph_op) {
            continue;
        }

        // cache for already constant folded inputs
        std::unordered_multimap<size_t, ConstantToPushDesc> cache;
        const size_t num_subgraphs = multi_sub_graph_op->get_internal_subgraphs_size();

        for (size_t body_idx = 0; body_idx < num_subgraphs; body_idx++) {
            auto& descriptions = multi_sub_graph_op->get_input_descriptions(body_idx);
            for (auto desc_it = descriptions.begin(); desc_it < descriptions.end(); ++desc_it) {
                const auto& desc = *desc_it;
                const auto input_index = desc->m_input_index;
                std::shared_ptr<ov::op::v0::Constant> constant = nullptr;
                bool used_in_many_bodies = false;
                const auto body_param_idx = desc->m_body_parameter_index;
                auto& body_param = multi_sub_graph_op->get_function(body_idx)->get_parameters()[body_param_idx];
                const auto it = cache.find(input_index);
                if (!std::dynamic_pointer_cast<MultiSubGraphOp::InvariantInputDescription>(desc)) {
                    continue;
                } else if (it != cache.end()) {
                    constant = it->second.m_constant;
                    // an input used earlier in the different body and not dangling
                    used_in_many_bodies = it->second.m_body_idx != body_idx &&
                                          body_param->get_output_target_inputs(0).size() > 0 &&
                                          it->second.m_body_param_consumers > 0;
                    it->second.m_used_in_many_bodies = used_in_many_bodies;
                } else {
                    constant = ov::util::constantfold_subgraph(op->input_value(input_index));
                    if (!constant) {
                        continue;
                    }
                }
                cache.insert({input_index,
                              ConstantToPushDesc(body_idx,
                                                 body_param_idx,
                                                 input_index,
                                                 constant,
                                                 body_param->get_output_target_inputs(0).size(),
                                                 used_in_many_bodies)});
            }
        }

        // cache for already constant folded inputs
        // bitmask describing which MultiSubGraphOp's input to remove
        std::vector<bool> remove_inputs_mask(multi_sub_graph_op->get_input_size(), false);
        for (const auto& it : cache) {
            const auto& const_to_push = it.second;
            if (!const_to_push.m_used_in_many_bodies) {
                const size_t body_idx = const_to_push.m_body_idx;
                const auto& body = multi_sub_graph_op->get_function(body_idx);
                auto& descriptions = multi_sub_graph_op->get_input_descriptions(body_idx);
                auto& body_param = body->get_parameters().at(const_to_push.m_body_param_idx);
                descriptions.erase(
                    std::remove_if(descriptions.begin(),
                                   descriptions.end(),
                                   [&const_to_push](const ov::op::util::MultiSubGraphOp::InputDescription::Ptr& in) {
                                       return in->m_body_parameter_index == const_to_push.m_body_param_idx &&
                                              in->m_input_index == const_to_push.m_op_in_idx;
                                   }));
                replace_body_parameter(body, body_param, const_to_push, descriptions, cache);
                remove_inputs_mask[const_to_push.m_op_in_idx] = true;
                result = true;
            }
        }

        if (result) {
            update_multi_sub_graph_op_inputs(multi_sub_graph_op, remove_inputs_mask);
        }

        for (size_t body_idx = 0; body_idx < num_subgraphs; body_idx++) {
            bool model_changed = run_on_model(multi_sub_graph_op->get_function(body_idx));
            result = result || model_changed;
        }
    }

    return result;
}
