// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/util/file_util.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/util/op_types.hpp"

#include "common_test_utils/file_utils.hpp"

#include "cache/cache.hpp"
#include "utils/memory.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {
size_t ICache::mem_size = get_ram_size();

bool ICache::serialize_model(const std::pair<std::shared_ptr<ov::Model>, MetaInfo>& graph_info,
                             const std::string& rel_serialization_dir) {
    std::shared_ptr<ov::Model> model = graph_info.first;
    MetaInfo meta = graph_info.second;

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
            ov::test::utils::removeFile(xml_path);
            ov::test::utils::removeFile(bin_path);
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