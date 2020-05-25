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
    // find '<net ' substring in the .xml file
    return (header.find("<net ") != std::string::npos) || (header.find("<Net ") != std::string::npos);
}

CNNNetwork IRReader::read(std::istream& model, const std::vector<IExtensionPtr>& exts) const {
    std::istringstream emptyStream;
    return read(model, emptyStream, exts);
}
CNNNetwork IRReader::read(std::istream& model, std::istream& weights, const std::vector<IExtensionPtr>& exts) const {
    model.seekg(0, model.beg);
    weights.seekg(0, weights.beg);
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

INFERENCE_ENGINE_READER_API(StatusCode) InferenceEngine::CreateReader(IReader*& reader, ResponseDesc *resp) noexcept {
    try {
        reader = new IRReader();
        return OK;
    }
    catch (std::exception &ex) {
        return GENERAL_ERROR;
    }
}
