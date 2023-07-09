// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/op_types.hpp"

#include "functional_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/graph_comparator.hpp"

#include "cache/graph_cache.hpp"
#include "utils/node.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

std::shared_ptr<GraphCache> GraphCache::m_cache_instance = nullptr;

void GraphCache::update_cache(const std::shared_ptr<ov::Model>& model, const std::string& model_meta_data, bool extract_body) {
    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(model);
    bool is_graph_started = false;
    std::map<std::string, std::shared_ptr<ov::Node>> model_vector;
    std::unordered_set<std::string> compiled_op_name;
    for (const auto& compiled_op : compiled_model.get_runtime_model()->get_ordered_ops()) {
        const auto& rt_info = compiled_op->get_rt_info();
        if (rt_info.count("originalLayersNames")) {
            compiled_op_name.insert(rt_info.find("originalLayersNames")->second.as<std::string>());
        }
    }

    for (const auto& op : model->get_ordered_ops()) {
        auto op_name = op->get_friendly_name();
        // std::cout << op_name << std::endl;
        if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op)) {
            continue;
        }
        if (op_name == "Multiply_8749") {
            auto h = 0;
        }
        auto cloned_op = clone_node(op, true, false, "Op_" + std::to_string(model_vector.size()));
        if (model_vector.empty()) {
            model_vector.insert({ op->get_friendly_name(), cloned_op });
        } else {
            ov::OutputVector out;
            out.resize(op->inputs().size());
            // std::cout << "DEBUG: " << out.size() << std::endl;
            for (size_t i = 0; i < op->inputs().size(); ++i) {
                auto in_node = op->get_input_node_ptr(i)->shared_from_this();
                auto in_node_cloned = cloned_op->get_input_node_ptr(i)->shared_from_this();
                for (size_t j = 0; j < in_node->outputs().size(); ++j) {
                    for (const auto& target_input : in_node->output(j).get_target_inputs()) {
                        auto out_in_node = target_input.get_node()->shared_from_this();
                        std::cout << op->get_friendly_name() << " " << in_node->get_friendly_name() << " " << out_in_node->get_friendly_name() << std::endl;
                        if (out_in_node == op) {
                            // std::cout << "DEBUG 2: " << in_node->get_friendly_name() << std::endl;
                            if (model_vector.count(in_node->get_friendly_name())) {
                                // std::cout << "DEBUG_1: " << j << std::endl;
                                // out[j] = out_in_node->output(j);
                                out[i] = model_vector.at(in_node->get_friendly_name())->output(j);
                            } else {
                                out[i] = cloned_op->get_input_node_ptr(i)->output(j);
                                auto g = 0;
                            }
                            break;
                        }
                    }
                }
            }
            model_vector.insert({ op->get_friendly_name(), cloned_op->clone_with_new_inputs(out)});
        }
        if (!compiled_op_name.count(op_name)) {
            if (model_vector.size() > 1) {
                ov::OutputVector results;
                std::map<std::string, InputInfo> input_info;
                for (const auto& op : model_vector) {
                    // auto a = op.second->outputs().begin()->get_target_inputs().begin()->get_node()->shared_from_this();
                    // std::cout << "DEBUG " << op.second->get_friendly_name() << " " << a->get_friendly_name() << std::endl;
                    auto this_input_info = get_input_info_by_node(op.second);
                    input_info.insert(this_input_info.begin(), this_input_info.end());
                    for (size_t j = 0; j < op.second->outputs().size(); ++j) {
                        if (op.second->output(j).get_target_inputs().empty()) {
                            results.push_back(std::make_shared<ov::op::v0::Result>(op.second->output(j)));
                        }
                    }
                }
                auto model = std::make_shared<ov::Model>(results);

                auto find_same_graph = [&model](const std::pair<std::shared_ptr<ov::Model>, MetaInfo>& cache_item){
                    auto ref_model = cache_item.first;
                    auto fc = FunctionsComparator::with_default()
                                .enable(FunctionsComparator::ATTRIBUTES)
                                .enable(FunctionsComparator::NODES)
                                .enable(FunctionsComparator::PRECISIONS)
                                .enable(FunctionsComparator::ATTRIBUTES)
                                .enable(FunctionsComparator::SUBGRAPH_DESCRIPTORS);
                    return fc.compare(model, ref_model).valid;
                };
                auto c = std::find_if(m_graph_cache.begin(), m_graph_cache.end(), find_same_graph);
                if (c != m_graph_cache.end()) {
                    auto ref_model_size = c->first->get_graph_size();
                    auto orig_model_size = model->get_graph_size();
                    if (orig_model_size < ref_model_size) {
                        auto meta = c->second;
                        meta.update(model_meta_data, input_info,
                        model->get_ops().size() - model->get_output_size() - model->inputs().size());
                        m_graph_cache.erase(c->first);
                        m_graph_cache.insert({model, meta});
                    } else {
                        c->second.update(model_meta_data, input_info,
                        model->get_ops().size() - model->get_output_size() - model->inputs().size());
                    }
                } else {
                    auto meta = MetaInfo(model_meta_data, input_info,
                                         model->get_ops().size() - model->get_output_size() - model->inputs().size());
                    // graph comparation
                    // add to cache smaller graph
                    std::cout << "DEBUG: " << model->get_ops().size() << std::endl;
                    m_graph_cache.insert({model, meta});
                    // serialize_model({model, meta}, "/Users/iefode/repo/temp/output_test/subgraph");
                    break;
                }
            }
            model_vector.clear();
        }
    }
    return;
}

void GraphCache::serialize_cache() {
    for (const auto& cache_item : m_graph_cache) {
        serialize_model(cache_item, "subgraph");
    }
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov