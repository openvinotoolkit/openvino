// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/loop.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/util/file_util.hpp"

#include "cache/op_cache.hpp"
#include "utils/node.hpp"
#include "op_conformance_utils/utils/file.hpp"

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
        if (ov::as_type_ptr<ov::op::v0::Parameter>(op) ||
            ov::as_type_ptr<ov::op::v0::Constant>(op) ||
            ov::as_type_ptr<ov::op::v0::Result>(op) ||
            // ReadValue and Assign have to be handled in pair
            // Will be handled as part of 48838
            ov::as_type_ptr<ov::op::util::AssignBase>(op) ||
            ov::as_type_ptr<ov::op::util::ReadValueBase>(op)) {
            continue;
        }
        if (extract_body) {
            if (ov::as_type_ptr<ov::op::v8::If>(op)) {
                auto if_op = ov::as_type_ptr<ov::op::v8::If>(op);
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    update_cache(if_body, model_path, extract_body, from_cache);
                }
            } else if (ov::as_type_ptr<ov::op::v5::Loop>(op)) {
                auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                update_cache(loop_body, model_path, extract_body, from_cache);
            } else if (ov::as_type_ptr<ov::op::v0::TensorIterator>(op)) {
                auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
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
    auto cloned_node = ov::util::clone_node(node, true);
    auto cloned_node_in_info = ov::util::get_input_info_by_node(cloned_node);
    bool in_info_is_matched = true;
    if (cloned_node == nullptr)
        return;
    // cloned_node->set_friendly_name(ov::test::functional::get_node_version(cloned_node));
    for (auto &&it : m_ops_cache) {
        in_info_is_matched = true;
        if (m_manager.match(it.first, cloned_node)) {
            // std::cout << "Match " << cloned_node->get_type_info().name <<  " " << cloned_node->get_friendly_name() <<
            //        " with " << it.first->get_friendly_name() << std::endl;
            for (const auto& in_info_item : it.second.get_input_info()) {
                if (!cloned_node_in_info.count(in_info_item.first)) {
                    in_info_is_matched = false;
                    break;
                }
                if (cloned_node_in_info[in_info_item.first].is_const != in_info_item.second.is_const) {
                    in_info_is_matched = false;
                    break;
                }
            }
            if (in_info_is_matched) {
                find_op_in_cache = it.first;
                break;
            }
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

    auto meta_path = ov::util::replace_extension(model_path, "meta");
    size_t priority = ov::util::get_node_priority_by_version(cloned_node);
    ov::conformance::MetaInfo meta = from_cache ? \
                                     ov::conformance::MetaInfo::read_meta_from_file(meta_path) : \
                                     ov::conformance::MetaInfo(model_path, cloned_node_in_info, model_op_cnt, 1,  "", priority);

    if (find_op_in_cache != nullptr) {
        // std::cout << "[ INFO ][ OP CACHE ] Update cache node: " << cloned_node->get_type_info().name << cloned_node->get_friendly_name() <<
        //     " " << find_op_in_cache->get_friendly_name() << std::endl;
        m_ops_cache[find_op_in_cache].update(model_path, cloned_node_in_info, model_op_cnt, 1, "", ignored_input_names);

        if (find_op_in_cache > cloned_node && in_info_is_matched) {
            auto old_meta = m_ops_cache[find_op_in_cache];
            m_ops_cache.erase(find_op_in_cache);
            m_ops_cache.insert({ cloned_node, old_meta });
        }
    } else {
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

bool
OpCache::serialize_op(const std::pair<std::shared_ptr<ov::Node>, ov::conformance::MetaInfo> &op_info) {
    std::string serialization_dir = get_rel_serilization_dir(op_info.first);
    std::shared_ptr<ov::Model> model = ov::util::generate_model_by_node(op_info.first);
    return serialize_model(make_pair(model, op_info.second), serialization_dir);
}

std::string OpCache::get_rel_serilization_dir(const std::shared_ptr<ov::Node>& node) {
    std::string op_folder_name = ov::util::get_node_version(node->get_type_info());
    auto op_el_type = node->get_output_element_type(0).get_type_name();
    return ov::util::path_join({m_cache_subdir, ov::util::get_node_type(node), op_folder_name, op_el_type}).string();
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
