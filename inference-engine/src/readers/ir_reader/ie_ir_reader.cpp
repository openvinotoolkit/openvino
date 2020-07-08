// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <xml_parse_utils.h>

#include <ie_ir_reader.hpp>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>

#include "description_buffer.hpp"
#include "ie_ir_parser.hpp"
#include "ie_ngraph_utils.hpp"

using namespace InferenceEngine;

static size_t GetIRVersion(pugi::xml_node& root) {
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

bool IRReader::supportModel(std::istream& model) const {
    model.seekg(0, model.beg);
    const int header_size = 128;
    std::string header(header_size, ' ');
    model.read(&header[0], header_size);
    model.seekg(0, model.beg);

    pugi::xml_document doc;
    auto res = doc.load_string(header.c_str(), pugi::parse_default | pugi::parse_fragment);

    bool supports = false;

    if (res == pugi::status_ok) {
        pugi::xml_node root = doc.document_element();

        std::string node_name = root.name();
        std::transform(node_name.begin(), node_name.end(), node_name.begin(), ::tolower);

        if (node_name == "net") {
            size_t const version = GetIRVersion(root);
#ifdef IR_READER_V10
            supports = version == 10;
#else
            supports = version < 10;
#endif
        }
    }

    return supports;
}

CNNNetwork IRReader::read(std::istream& model, const std::vector<IExtensionPtr>& exts) const {
    std::istringstream emptyStream;
    return read(model, emptyStream, exts);
}

CNNNetwork IRReader::read(std::istream& model, std::istream& weights, const std::vector<IExtensionPtr>& exts) const {
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load(model);
    if (res.status != pugi::status_ok) {
        THROW_IE_EXCEPTION << res.description() << "at offset " << res.offset;
    }
    pugi::xml_node root = xmlDoc.document_element();

    auto version = GetIRVersion(root);
    IRParser parser(version, exts);
    return CNNNetwork(parser.parse(root, weights));
}

INFERENCE_PLUGIN_API(StatusCode) InferenceEngine::CreateReader(IReader*& reader, ResponseDesc *resp) noexcept {
    try {
        reader = new IRReader();
        return OK;
    }
    catch (std::exception &) {
        return GENERAL_ERROR;
    }
}
