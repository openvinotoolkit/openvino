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

    for (auto && node : f->get_ordered_ops()) {
        uint64_t port_index{node->inputs().size()};
        for (auto && output : node->outputs()) {
            const auto & t = output.get_tensor_ptr();
            for (const auto & port_name : t->get_names()) {
                auto map_node = root_node.append_child("map");
                auto framework_node = map_node.append_child("framework");
                auto ir_node = map_node.append_child("IR");

                framework_node.append_attribute("name").set_value(node->get_friendly_name().c_str());
                framework_node.append_attribute("output_port_id").set_value(port_name.c_str());

                ir_node.append_attribute("name").set_value(node->get_friendly_name().c_str());
                ir_node.append_attribute("output_port_id").set_value(port_index);
            }
            ++port_index;
        }
    }

    // save mapping file
    std::ofstream mapping_file(m_path_to_file, std::ios::out);
    xml_doc.save(mapping_file);
    mapping_file.flush();
    return false;
}