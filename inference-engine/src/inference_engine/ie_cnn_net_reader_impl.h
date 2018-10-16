// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_icnn_net_reader.h"
#include "cnn_network_impl.hpp"
#include <memory>
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
    virtual std::shared_ptr<IFormatParser> create(int version) = 0;
};

struct V2FormatParserCreator : public FormatParserCreator {
    std::shared_ptr<IFormatParser> create(int version) override;
};

class CNNNetReaderImpl : public ICNNNetReader {
public:
    static std::string NameFromFilePath(const char *filepath);

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
        strncpy(name, this->name.c_str(), len - 1);
        if (len) name[len-1] = '\0';  // strncpy is not doing this, so output might be not null-terminated
        return OK;
    }

    int getVersion(ResponseDesc * resp) noexcept override {
        return version;
    }

    void Release() noexcept override {
        delete this;
    }

private:
    std::shared_ptr<InferenceEngine::details::IFormatParser> _parser;

    static int GetFileVersion(pugi::xml_node &root);

    StatusCode ReadNetwork(pugi::xml_document &xmlDoc);

    std::string description;
    std::string name;
    InferenceEngine::details::CNNNetworkImplPtr network;
    bool parseSuccess;
    int version;
    FormatParserCreator::Ptr parserCreator;
};
}  // namespace details
}  // namespace InferenceEngine
