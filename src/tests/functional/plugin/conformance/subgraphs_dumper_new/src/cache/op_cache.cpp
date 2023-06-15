// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>

// #include "op_cloner.hpp"
#include "openvino/core/core.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/util/file_util.hpp"

#include "cache/op_cache.hpp"
#include "utils/node.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {
std::shared_ptr<OpCache> OpCache::m_cache_instance = nullptr;

void OpCache::update_cache(const std::shared_ptr<ov::Model>& model,
                           const std::string& model_path,
                           bool extract_body) {
    size_t model_op_cnt = model->get_ops().size() - model->get_output_size() - model->inputs().size();
    for (const auto& op : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(op) ||
            std::dynamic_pointer_cast<ov::op::v0::Constant>(op) ||
            std::dynamic_pointer_cast<ov::op::v0::Result>(op) ||
            // ReadValue and Assign have to be handled in pair
            // Will be handled as part of 48838
            std::dynamic_pointer_cast<ov::op::util::AssignBase>(op) ||
            std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(op)) {
            continue;
        }
        if (extract_body) {
            if (std::dynamic_pointer_cast<ov::op::v8::If>(op)) {
                auto if_op = std::dynamic_pointer_cast<ov::op::v8::If>(op);
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    update_cache(if_body, model_path, extract_body);
                }
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                update_cache(loop_body, model_path, extract_body);
            } else if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
                auto ti = std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                update_cache(ti_body, model_path, extract_body);
            }
        }
        update_cache(op, model_path, model_op_cnt);
    }
}

void OpCache::update_cache(const std::shared_ptr<ov::Node>& node,
                           const std::string& model_path,
                           size_t model_op_cnt) {
    std::shared_ptr<ov::Node> find_op_in_cache = nullptr;
    for (auto &&it : m_ops_cache) {
        if (m_manager.match_any(it.first, node)) {
            it.second.update(model_path, get_input_info_by_node(node), model_op_cnt);
            find_op_in_cache = it.first;
            break;
        }
    }

    auto meta = MetaInfo(model_path, get_input_info_by_node(node), model_op_cnt);
    if (find_op_in_cache > node) {
        meta = m_ops_cache[find_op_in_cache];
        m_ops_cache.erase(find_op_in_cache);
        find_op_in_cache = nullptr;
    }
    if (find_op_in_cache == nullptr) {
        m_ops_cache.insert({ node, meta });
    }
}

void OpCache::serialize_cache() {
    for (const auto& cache_item : m_ops_cache) {
        serialize_op(cache_item);
    }
}

bool OpCache::serialize_op(const std::pair<std::shared_ptr<ov::Node>, MetaInfo> &op_info) {
    std::string serialization_dir = get_rel_serilization_dir(op_info.first);
    std::shared_ptr<ov::Model> model = generate_graph_by_node(op_info.first);
    model->set_friendly_name(op_info.first->get_friendly_name());
    return serialize_model(make_pair(model, op_info.second), serialization_dir);
}

std::string OpCache::get_rel_serilization_dir(const std::shared_ptr<ov::Node>& node) {
    std::string op_folder_name = node->get_type_info().name;
    std::string opset_version = node->get_type_info().get_version();
    std::string opset_name = "opset";
    auto pos = opset_version.find(opset_name);
    if (pos != std::string::npos) {
        op_folder_name += "-" + opset_version.substr(pos + opset_name.size());
    }
    auto op_el_type = node->get_element_type().get_type_name();

    return ov::util::path_join({"operation", get_node_type(node), op_folder_name, op_el_type});
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
