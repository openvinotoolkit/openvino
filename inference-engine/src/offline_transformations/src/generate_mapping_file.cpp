// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ostream>
#include <fstream>

#include "generate_mapping_file.hpp"

#include "pugixml.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::GenerateMappingFile, "GenerateMappingFile", 0);

bool ngraph::pass::GenerateMappingFile::run_on_function(std::shared_ptr<ngraph::Function> f) {
    pugi::xml_document xml_doc;
    pugi::xml_node root_node = xml_doc.append_child("mapping");

    auto add_mapping = [&](const std::string & fw_name, const std::string & fw_port_name,
                           const std::string & ir_name, const std::string & ir_port_name) {
        auto map_node = root_node.append_child("map");
        auto framework_node = map_node.append_child("framework");
        auto ir_node = map_node.append_child("IR");

        framework_node.append_attribute("name").set_value(fw_name.c_str());
        framework_node.append_attribute("output_port_id").set_value(fw_port_name.c_str());

        ir_node.append_attribute("name").set_value(ir_name.c_str());
        ir_node.append_attribute("output_port_id").set_value(ir_port_name.c_str());
    };

    auto extract_name = [](const std::string & port_name) -> std::string {
        return port_name.substr(0, port_name.find(':'));
    };

    for (auto && node : f->get_ordered_ops()) {
        uint64_t ie_port_index{node->inputs().size()};
        uint64_t ng_port_index{0};
        for (auto && output : node->outputs()) {
            const auto & node_name = node->get_friendly_name();
            const auto & t = output.get_tensor_ptr();

            for (const auto & port_name : t->get_names()) {
                add_mapping(node_name, port_name, node_name, std::to_string(ie_port_index));

                if (m_extract_name) {
                    for (auto &name : t->get_names()) {
                        add_mapping(extract_name(name), port_name, node_name, std::to_string(ie_port_index));
                    }
                }
            }
            ++ie_port_index;
            ++ng_port_index;
        }
    }

    // save mapping file
    std::ofstream mapping_file(m_path_to_file, std::ios::out);
    xml_doc.save(mapping_file);
    mapping_file.flush();
    return false;
}