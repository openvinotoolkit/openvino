// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <xml_parse_utils.h>

#include <array>
#include <fstream>

namespace InferenceEngine {
namespace details {

inline size_t get_ir_version(pugi::xml_node& root) {
    return pugixml::utils::GetUIntAttr(root, "version", 0);
}

/**
 * @brief Extracts IR version from model stream
 * @param model Models stream
 * @return IR version, 0 if model does represent IR
 */
inline size_t get_ir_version(std::istream& model) {
    std::array<char, 512> header = {};

    model.seekg(0, model.beg);
    model.read(header.data(), header.size());
    model.clear();
    model.seekg(0, model.beg);

    pugi::xml_document doc;
    auto res =
        doc.load_buffer(header.data(), header.size(), pugi::parse_default | pugi::parse_fragment, pugi::encoding_utf8);

    if (res == pugi::status_ok) {
        pugi::xml_node root = doc.document_element();

        std::string node_name = root.name();
        std::transform(node_name.begin(), node_name.end(), node_name.begin(), ::tolower);

        if (node_name == "net") {
            return get_ir_version(root);
        }
    }

    return 0;
}

}  // namespace details
}  // namespace InferenceEngine
