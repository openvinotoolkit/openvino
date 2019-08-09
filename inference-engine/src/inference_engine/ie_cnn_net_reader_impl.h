// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_icnn_net_reader.h"
#include "ie_memcpy.h"
#include "cnn_network_impl.hpp"
#include "parsers.h"
#include <memory>
#include <algorithm>
#include <string>
#include <map>

namespace pugi {
class xml_node;

class xml_document;
}  // namespace pugi

namespace InferenceEngine {
namespace details {

struct FormatParserCreator {
    using Ptr = std::shared_ptr<FormatParserCreator>;
    virtual std::shared_ptr<IFormatParser> create(size_t version) = 0;
};

struct V2FormatParserCreator : public FormatParserCreator {
    std::shared_ptr<IFormatParser> create(size_t version) override;
};

class CNNNetReaderImpl : public ICNNNetReader {
public:
    explicit CNNNetReaderImpl(const FormatParserCreator::Ptr& _parserCreator);

    StatusCode ReadNetwork(const char *filepath, ResponseDesc *resp) noexcept override;

    StatusCode ReadNetwork(const void *model, size_t size, ResponseDesc *resp)noexcept override;

    StatusCode SetWeights(const TBlob<uint8_t>::Ptr &weights, ResponseDesc *resp) noexcept override;

    StatusCode ReadWeights(const char *filepath, ResponseDesc *resp) noexcept override;

    ICNNNetwork *getNetwork(ResponseDesc *resp) noexcept override {
        return network.get();
    }

    bool isParseSuccess(ResponseDesc *resp) noexcept override {
        return parseSuccess;
    }

    StatusCode getDescription(ResponseDesc *desc) noexcept override {
        return DescriptionBuffer(OK, desc) << description;
    }

    StatusCode getName(char *name, size_t len, ResponseDesc *resp) noexcept override {
        if (len > 0) {
            size_t length = std::min(this->name.size(), len-1);  // cut the name if buffer is too small
            ie_memcpy(name, len, this->name.c_str(), length);
            name[length] = '\0';  // null terminate
        }
        return OK;
    }

    int getVersion(ResponseDesc * resp) noexcept override {
        return _version;
    }

    void Release() noexcept override {
        delete this;
    }

private:
    std::shared_ptr<InferenceEngine::details::IFormatParser> _parser;
    size_t GetFileVersion(pugi::xml_node &root);
    StatusCode ReadNetwork(pugi::xml_document &xmlDoc);

    std::string description;
    std::string name;
    InferenceEngine::details::CNNNetworkImplPtr network;
    bool parseSuccess;
    size_t _version;
    FormatParserCreator::Ptr parserCreator;
};
}  // namespace details
}  // namespace InferenceEngine
