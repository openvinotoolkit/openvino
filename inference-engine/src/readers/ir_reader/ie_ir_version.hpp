
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <xml_parse_utils.h>

namespace InferenceEngine {
namespace details {

inline size_t GetIRVersion(pugi::xml_node& root) {
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

/**
 * @brief Extracts IR version from model stream
 * @param model Models stream
 * @return IR version, 0 if model does represent IR
 */
size_t GetIRVersion(std::istream& model) {
    model.seekg(0, model.beg);
    const int header_size = 128;
    std::string header(header_size, ' ');
    model.read(&header[0], header_size);
    model.seekg(0, model.beg);

    pugi::xml_document doc;
    auto res = doc.load_string(header.c_str(), pugi::parse_default | pugi::parse_fragment);

    if (res == pugi::status_ok) {
        pugi::xml_node root = doc.document_element();

        std::string node_name = root.name();
        std::transform(node_name.begin(), node_name.end(), node_name.begin(), ::tolower);

        if (node_name == "net") {
            return GetIRVersion(root);
        }
    }

    return 0;
}

}  // namespace details
}  // namespace InferenceEngine