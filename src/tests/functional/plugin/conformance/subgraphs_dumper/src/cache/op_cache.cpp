// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>

#include "openvino/core/core.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/util/file_util.hpp"

#include "common_test_utils/file_utils.hpp"

#include "cache/op_cache.hpp"
#include "utils/node.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {
std::shared_ptr<OpCache> OpCache::m_cache_instance = nullptr;

void OpCache::update_cache(const std::shared_ptr<ov::Model>& model,
                           const std::string& model_path,
                           bool extract_body, bool from_cache) {
    std::cout << "[ INFO ][ OP CACHE ] Processing model: " << model_path << std::endl;
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
                    update_cache(if_body, model_path, extract_body, from_cache);
                }
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                update_cache(loop_body, model_path, extract_body, from_cache);
            } else if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
                auto ti = std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                update_cache(ti_body, model_path, extract_body, from_cache);
            }
        }
        update_cache(op, model_path, model_op_cnt, from_cache);
    }
}

void OpCache::update_cache(const std::shared_ptr<ov::Node>& node,
                           const std::string& model_path,
                           size_t model_op_cnt, bool from_cache) {
    std::shared_ptr<ov::Node> find_op_in_cache = nullptr;
    // Clone node to get node with Parameter/Constants input only
    auto cloned_node = clone_node(node, true);
    if (cloned_node == nullptr)
        return;
    cloned_node->set_friendly_name(ov::test::functional::get_node_version(cloned_node));
    for (auto &&it : m_ops_cache) {
        if (m_manager.match(it.first, cloned_node)) {
            // std::cout << "Match " << cloned_node->get_type_info().name <<  " " << cloned_node->get_friendly_name() <<
            //        " with " << it.first->get_friendly_name() << std::endl;
            find_op_in_cache = it.first;
            break;
        }
    }

    // to identify ignored inputs
    std::vector<std::string> ignored_input_names = {};
    {
        auto matching_config = m_manager.get_config(find_op_in_cache);
        if (matching_config) {
            for (const auto& ignored_port : matching_config->ignored_ports) {
                ignored_input_names.push_back(find_op_in_cache->get_friendly_name() + "_" + std::to_string(ignored_port));
            }
        }
    }

    MetaInfo meta;
    if (from_cache) {
        auto meta_path = ov::test::utils::replaceExt(model_path, "meta");
        meta = MetaInfo::read_meta_from_file(meta_path);
    } else {
        size_t priority = get_node_priority_by_version(cloned_node);
        meta = MetaInfo(model_path, get_input_info_by_node(cloned_node), model_op_cnt, 1,  "", priority);
    }

    if (find_op_in_cache != nullptr) {
        // std::cout << "[ INFO ][ OP CACHE ] Update cache node: " << cloned_node->get_type_info().name << cloned_node->get_friendly_name() <<
        //     " " << find_op_in_cache->get_friendly_name() << std::endl;
        m_ops_cache[find_op_in_cache].update(
            model_path, get_input_info_by_node(cloned_node), model_op_cnt, 1, "", ignored_input_names);
    }

    if (find_op_in_cache > cloned_node) {
        meta = m_ops_cache[find_op_in_cache];
        m_ops_cache.erase(find_op_in_cache);
        find_op_in_cache = nullptr;
    }

    if (find_op_in_cache == nullptr) {
        // std::cout << "[ INFO ][ OP CACHE ] Insert node: " << cloned_node->get_type_info().name <<
        //     " " << cloned_node->get_friendly_name() << " to Cache" << std::endl;
        m_ops_cache.insert({ cloned_node, meta });
    }
}

void OpCache::serialize_cache() {
    for (const auto& cache_item : m_ops_cache) {
        serialize_op(cache_item);
    }
}

bool OpCache::serialize_op(const std::pair<std::shared_ptr<ov::Node>, MetaInfo> &op_info) {
    std::string serialization_dir = get_rel_serilization_dir(op_info.first);
    std::shared_ptr<ov::Model> model = generate_model_by_node(op_info.first);
    return serialize_model(make_pair(model, op_info.second), serialization_dir);
}

std::string OpCache::get_rel_serilization_dir(const std::shared_ptr<ov::Node>& node) {
    std::string op_folder_name = ov::test::functional::get_node_version(node);
    auto op_el_type = node->get_output_element_type(0).get_type_name();

    return ov::util::path_join({m_cache_subdir, get_node_type(node), op_folder_name, op_el_type});
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
