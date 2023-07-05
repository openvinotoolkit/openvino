// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/op_types.hpp"

#include "functional_test_utils/ov_plugin_cache.hpp"

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
    std::vector<std::shared_ptr<ov::Node>> model_vector;
    for (const auto& smt : compiled_model.get_runtime_model()->get_ordered_ops()) {
        std::string b = "";
        for (const auto& aa : smt->get_rt_info()) {
            // add body handling
            std::cout << aa.first << " " << aa.second.as<std::string>() << std::endl;
            if (aa.first == "originalLayersNames") {
                b = aa.second.as<std::string>();
            }
        }
        if (b != smt->get_friendly_name()) {
            if (!is_graph_started) {
                is_graph_started = true;
                model_vector.push_back(clone_node(smt));
            } else {
                model_vector.push_back(smt->clone_with_new_inputs({model_vector.back()->outputs()}));
            }
        } else if (is_graph_started) {
            if (model_vector.size() > 1) {
                ov::OutputVector results;
                for (auto& out : model_vector.back()->outputs()) {
                    results.push_back(std::make_shared<ov::op::v0::Result>(out));
                }
                auto model = std::make_shared<ov::Model>(results);
                auto meta = MetaInfo(model_meta_data, get_input_info_by_node(model_vector.front()),
                                    model->get_ops().size() - model->get_output_size() - model->inputs().size());
                // graph comparation
                // add to cache smaller graph
                m_graph_cache.insert({model, meta});
            }
            is_graph_started = false;
            model_vector.clear();
        }
        std::cout << "smt" << std::endl;
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