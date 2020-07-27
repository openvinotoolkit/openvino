// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <xml_parse_utils.h>

#include <ie_ir_version.hpp>
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

bool IRReader::supportModel(std::istream& model) const {
    auto version = details::GetIRVersion(model);

#ifdef IR_READER_V10
    return version == 10;
#else
    return version > 1 && version <= 7;
#endif
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

    auto version = details::GetIRVersion(root);
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
