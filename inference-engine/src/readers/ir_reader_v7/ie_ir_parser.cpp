// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_reader.hpp"
#include "ie_ir_parser.hpp"
#include "ie_cnn_net_reader_impl.h"

using namespace InferenceEngine;

IRParser::IRParser(size_t version): IRParser(version, {}) {}
IRParser::IRParser(size_t version, const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    if (version < 10) {
        parser = std::make_shared<CNNParser>();
        return;
    } else {
        IE_THROW() << "Unsupported IR version: " << version;
    }
}

CNNNetwork IRParser::parse(const pugi::xml_node& root, const Blob::CPtr& weights) {
    return parser->parse(root, weights);
}

/**
 * Hold original blob in order to avoid situations when original blob is allocated on stack
 */
class WeightsHolderBlob : public TBlob<uint8_t> {
    Blob::CPtr originBlob;

public:
    explicit WeightsHolderBlob(const Blob::CPtr& weights) :
        TBlob<uint8_t>(weights->getTensorDesc(),
                       weights->cbuffer().as<uint8_t*>()),
        originBlob(weights) { }
};

CNNNetwork CNNParser::parse(const pugi::xml_node& root, const Blob::CPtr& weights) {
    details::CNNNetReaderImpl reader(std::make_shared<details::V2FormatParserCreator>());
    ResponseDesc resp;
    StatusCode ret = reader.ReadNetwork(root, &resp);
    if (ret != OK)
        IE_THROW() << resp.msg;

    TBlob<uint8_t>::Ptr weightsPtr;

    if (weights != nullptr) {
        weightsPtr = TBlob<uint8_t>::Ptr(new WeightsHolderBlob(weights));
    } else {
        weightsPtr = std::make_shared<TBlob<uint8_t>>(TensorDesc(Precision::U8, { 0 }, Layout::C));
        weightsPtr->allocate();
    }
    ret = reader.SetWeights(weightsPtr, &resp);
    if (ret != OK)
        IE_THROW() << resp.msg;
    return reader.getNetwork();
}
