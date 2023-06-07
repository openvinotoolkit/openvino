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

namespace ov {
namespace tools {
namespace subgraph_dumper {
std::shared_ptr<OpCache> OpCache::m_cache_instance = nullptr;

void OpCache::update_cache(const std::shared_ptr<ov::Model>& model, const std::string& model_meta_data, bool extract_body) {
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
                    update_cache(if_body, model_meta_data, extract_body);
                }
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                update_cache(loop_body, model_meta_data, extract_body);
            } else if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
                auto ti = std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                update_cache(ti_body, model_meta_data, extract_body);
            }
        }
        update_cache(op, model_meta_data);
    }
}

// todo: iefode: check the function
void OpCache::update_cache(const std::shared_ptr<ov::Node>& node, const std::string& op_meta_data) {
    // const std::shared_ptr<ov::Node> cachedOp = [&] {
    //     for (auto &&it : m_ops_cache) {
    //         if (m_manager.match_any(it.first, node, it.second)) {
    //             it.second.found_in_models[op_meta_data.name].unique_op_cnt++;
    //             it.second.found_in_models[op_meta_data.name].model_paths.insert({{op_meta_data.path, op_meta_data.op_cnt}});
    //             return it.first;
    //         }
    //     }
    //     return std::shared_ptr<ov::Node>{};
    // }();

    // auto saveOpToCash = [&] {
    //     try {
    //         const auto& clone_fn = SubgraphsDumper::ClonersMap::cloners.at(node->get_type_info());
    //         LayerTestsUtils::MetaInfo meta(op_meta_data.name, op_meta_data.path, op_meta_data.op_cnt);
    //         const std::shared_ptr<ov::Node> op_clone = clone_fn(node, meta);
    //         if (!op_clone) {
    //             return;
    //         }
    //         op_clone->set_friendly_name(op_clone->get_friendly_name() + "_cached");
    //         m_ops_cache.insert({op_clone, meta});
    //     } catch (std::out_of_range& e) {
    //         std::cout << "WARNING: Cloner for " << node->get_type_name() << " (" << node->get_type_info().get_version()
    //                   << ") isn't found: " << e.what() << std::endl;
    //     } catch (std::exception& e) {
    //         std::cout << "ERROR: " << e.what() << std::endl;
    //     }
    // };

    // if (!cachedOp.get()) {
    //     saveOpToCash();
    // } else {
    //     for (int i = 0; i < node->get_input_size(); i++) {
    //         auto shape = node->get_input_shape(i);
    //         unsigned long shapeSize = ov::shape_size(shape) * node->get_output_element_type(0).size();

    //         auto cachedOpShape = cachedOp->get_input_shape(i);
    //         unsigned long cachedOpShapeSize =
    //             ov::shape_size(cachedOpShape) * cachedOp->get_output_element_type(0).size();

    //         if (shapeSize < cachedOpShapeSize) {
    //             m_ops_cache.erase(cachedOp);
    //             saveOpToCash();
    //         }
    //     }
    // }
}

void OpCache::serialize_cache() {
    for (const auto& cache_item : m_ops_cache) {
        serialize_op(cache_item);
    }
}

bool OpCache::serialize_op(const std::pair<std::shared_ptr<ov::Node>, MetaInfo> &op_info) {
    std::string serialization_dir = get_rel_serilization_dir(op_info.first);
    std::shared_ptr<ov::Model> model = generate_graph_by_node(op_info.first);
    return serialize_model(make_pair(model, op_info.second), serialization_dir);
}

std::shared_ptr<ov::Model> OpCache::generate_graph_by_node(const std::shared_ptr<ov::Node>& node) {
    ov::ParameterVector params;
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        if (ov::op::util::is_parameter(node->get_input_node_ptr(i))) {
            auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(
                    node->get_input_node_shared_ptr(i));
            params.push_back(param);
        }
    }
    ov::ResultVector results;
    for (auto &out : node->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(out));
    }
    return std::make_shared<ov::Model>(results, params);
}

std::string node_type(const std::shared_ptr<ov::Node>& node) {
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        if (node->get_input_partial_shape(i).is_dynamic()) {
            return "dynamic";
        }
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return "dynamic";
        }
    }
    return "static";
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

    return ov::util::path_join({"operation", node_type(node), op_folder_name, op_el_type});
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
