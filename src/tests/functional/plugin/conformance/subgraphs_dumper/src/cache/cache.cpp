// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/util/file_util.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/util/op_types.hpp"

#include "common_test_utils/file_utils.hpp"

#include "cache/cache.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

bool ICache::serialize_model(const std::pair<std::shared_ptr<ov::Model>, MetaInfo>& graph_info,
                             const std::string& rel_serialization_dir) {
    std::shared_ptr<ov::Model> model = graph_info.first;
    MetaInfo meta = graph_info.second;
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> nodes;
    ov::ParameterVector param_vector;
    for (const auto& op : model->get_ordered_ops()) {
        std::shared_ptr<ov::op::v0::Parameter> param = nullptr;
        if (ov::op::util::is_parameter(op)) {
            param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(op);
        } else if (ov::op::util::is_constant(op)) {
            auto op_to_replace = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
            if (op_to_replace->get_byte_size() > 1024) {
                param = std::make_shared<ov::op::v0::Parameter>(
                    op_to_replace->get_output_element_type(0), op_to_replace->get_output_partial_shape(0));
                param->set_friendly_name(op_to_replace->get_friendly_name());
                nodes.insert({ op_to_replace, param });
            }
        }
        if (param != nullptr) {
            param_vector.push_back(param);
        }
    }
    if (!nodes.empty()) {
        for (const auto& node : nodes) {
            model->replace_node(node.first, node.second);
        }
        model = std::make_shared<ov::Model>(model->get_results(), param_vector);
    }

    std::string model_name = model->get_friendly_name();
    std::string abs_searilization_dir = ov::util::path_join({ m_serialization_dir, rel_serialization_dir });
    std::string xml_path =  ov::util::path_join({ abs_searilization_dir, model_name + ".xml" });
    std::string bin_path = ov::util::path_join({ abs_searilization_dir, model_name + ".bin" });
    std::string meta_path = ov::util::path_join({ abs_searilization_dir, model_name + ".meta" });

    if (!ov::util::directory_exists(abs_searilization_dir)) {
        ov::util::create_directory_recursive(abs_searilization_dir);
    }
    auto exit_time = std::chrono::system_clock::now() + std::chrono::seconds(m_serialization_timeout);
    do {
        try {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>(xml_path, bin_path);
            manager.run_passes(model);
            model->validate_nodes_and_infer_types();
            meta.serialize(meta_path);
            return true;
        } catch (std::exception &e) {
            std::cout << "[ ERROR ] Failed to serialize model: " << model_name
                        << ". Exception: " << e.what() << std::endl;
            ov::test::utils::removeIRFiles(xml_path, bin_path);
            ov::test::utils::removeFile(meta_path);
            if (std::string(e.what()).find("Can't open") == std::string::npos) {
                return false;
            }
        }
    } while (std::chrono::system_clock::now() < exit_time);
    return false;
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov