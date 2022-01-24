// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <xml_parse_utils.h>

#include "openvino/runtime/common.hpp"
#include <legacy/ie_ir_version.hpp>
#include <ie_ir_reader.hpp>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

#include "ie_ir_parser.hpp"
#include "ie_ir_itt.hpp"

using namespace InferenceEngine;

bool IRReader::supportModel(std::istream& model) const {
    OV_ITT_SCOPED_TASK(itt::domains::V7Reader, "IRReader::supportModel");
    auto version = details::GetIRVersion(model);
    return version > 1 && version <= 7;
}

CNNNetwork IRReader::read(std::istream& model, const std::vector<IExtensionPtr>& exts) const {
    return read(model, nullptr, exts);
}

CNNNetwork IRReader::read(std::istream& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) const {
    OV_ITT_SCOPED_TASK(itt::domains::V7Reader, "IRReader::read");
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load(model);
    if (res.status != pugi::status_ok) {
        IE_THROW() << res.description() << " at offset " << res.offset;
    }
    pugi::xml_node root = xmlDoc.document_element();

    auto version = details::GetIRVersion(root);
    IRParser parser(version, exts);
    return CNNNetwork(parser.parse(root, weights));
}

INFERENCE_PLUGIN_API(void) InferenceEngine::CreateReader(std::shared_ptr<IReader>& reader) {
    reader = std::make_shared<IRReader>();
}
