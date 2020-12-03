// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <description_buffer.hpp>
#include <ie_cnn_net_reader_impl.h>
#include <ie_blob_stream.hpp>

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ie_format_parser.h"
#include "ie_ir_itt.hpp"
#include "parsers.h"
#include "xml_parse_utils.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

CNNNetReaderImpl::CNNNetReaderImpl(const FormatParserCreator::Ptr& _creator)
    : parseSuccess(false), _version(0), parserCreator(_creator) {}

StatusCode CNNNetReaderImpl::SetWeights(const TBlob<uint8_t>::Ptr& weights, ResponseDesc* desc) noexcept {
    if (!_parser && _version < 10) {
        return DescriptionBuffer(desc) << "network must be read first";
    }

    try {
        if (_version < 10) {
            _parser->SetWeights(weights);
        }
    } catch (const InferenceEngineException& iee) {
        xmlDoc.reset();
        return DescriptionBuffer(desc) << iee.what();
    }

    xmlDoc.reset();
    return OK;
}

static size_t GetFileVersion(pugi::xml_node& root) {
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

StatusCode CNNNetReaderImpl::ReadNetwork(const void* model, size_t size, ResponseDesc* resp) noexcept {
    if (network) {
        return DescriptionBuffer(NETWORK_NOT_READ, resp)
               << "Network has been read already, use new reader instance to read new network.";
    }

    xmlDoc = std::make_shared<pugi::xml_document>();
    pugi::xml_parse_result res = xmlDoc->load_buffer(model, size);
    if (res.status != pugi::status_ok) {
        return DescriptionBuffer(resp) << res.description() << "at offset " << res.offset;
    }
    StatusCode ret = ReadNetwork();
    if (ret != OK) {
        return DescriptionBuffer(resp) << "Error reading network: " << description;
    }
    return OK;
}

namespace {

void readAllFile(const std::string& string_file_name, void* buffer, size_t maxSize) {
    std::ifstream inputFile;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring file_name = FileUtils::multiByteCharToWString(string_file_name.c_str());
#else
    std::string file_name = string_file_name;
#endif

    inputFile.open(file_name, std::ios::binary | std::ios::in);
    if (!inputFile.is_open()) THROW_IE_EXCEPTION << "cannot open file " << string_file_name;
    if (!inputFile.read(reinterpret_cast<char*>(buffer), maxSize)) {
        inputFile.close();
        THROW_IE_EXCEPTION << "cannot read " << maxSize << " bytes from file " << string_file_name;
    }

    inputFile.close();
}

}  // namespace

StatusCode CNNNetReaderImpl::ReadWeights(const char* filepath, ResponseDesc* resp) noexcept {
    OV_ITT_SCOPED_TASK(itt::domains::V7Reader, "CNNNetReaderImpl::ReadWeights");
    int64_t fileSize = FileUtils::fileSize(filepath);

    if (fileSize < 0)
        return DescriptionBuffer(resp) << "filesize for: " << filepath << " - " << fileSize
                                       << "<0. Please, check weights file existence.";

    // If IR V10 then there hasn't been loaded network yet
    if (network.get() == nullptr && _version < 10) {
        return DescriptionBuffer(resp) << "network is empty";
    }

    auto ulFileSize = static_cast<size_t>(fileSize);

    try {
        TBlob<uint8_t>::Ptr weightsPtr(new TBlob<uint8_t>(TensorDesc(Precision::U8, {ulFileSize}, Layout::C)));
        weightsPtr->allocate();
        readAllFile(filepath, weightsPtr->buffer(), ulFileSize);
        return SetWeights(weightsPtr, resp);
    } catch (const InferenceEngineException& ex) {
        return DescriptionBuffer(resp) << ex.what();
    }
}

ICNNNetwork* CNNNetReaderImpl::getNetwork(ResponseDesc* resp) noexcept {
    OV_ITT_SCOPED_TASK(itt::domains::V7Reader, "CNNNetReaderImpl::getNetwork");
    return network.get();
}

StatusCode CNNNetReaderImpl::ReadNetwork(const char* filepath, ResponseDesc* resp) noexcept {
    OV_ITT_SCOPED_TASK(itt::domains::V7Reader, "CNNNetReaderImpl::ReadNetwork");
    if (network) {
        return DescriptionBuffer(NETWORK_NOT_READ, resp)
               << "Network has been read already, use new reader instance to read new network.";
    }

    auto parse_result = ParseXml(filepath);
    if (!parse_result.error_msg.empty()) {
        return DescriptionBuffer(resp) << parse_result.error_msg;
    }
    xmlDoc = std::move(parse_result.xml);

    StatusCode ret = ReadNetwork();
    if (ret != OK) {
        return DescriptionBuffer(resp) << "Error reading network: " << description;
    }
    return OK;
}

StatusCode CNNNetReaderImpl::ReadNetwork(const pugi::xml_node& const_root, ResponseDesc * desc) {
    try {
        pugi::xml_node root = *const_cast<pugi::xml_node*>(&const_root);
        _version = GetFileVersion(root);
        if (_version < 2) THROW_IE_EXCEPTION << "deprecated IR version: " << _version;

        if (_version < 10) {
            _parser = parserCreator->create(_version);
            InferenceEngine::details::CNNNetworkImplPtr local_network = _parser->Parse(root);
            name = local_network->getName();
            local_network->validate(static_cast<int>(_version));
            network = local_network;
            parseSuccess = true;
        } else {
            THROW_IE_EXCEPTION << "cannot parse future versions: " << _version;
        }
    } catch (const std::string& err) {
        parseSuccess = false;
        return DescriptionBuffer(desc) << err;
    } catch (const InferenceEngineException& e) {
        description = e.what();
        parseSuccess = false;
        return DescriptionBuffer(desc) << e.what();
    } catch (const std::exception& e) {
        description = e.what();
        parseSuccess = false;
        return DescriptionBuffer(desc) << e.what();
    } catch (...) {
        parseSuccess = false;
        return DescriptionBuffer(UNEXPECTED, desc) << "Unknown exception thrown";
    }

    return OK;
}

StatusCode CNNNetReaderImpl::ReadNetwork() {
    description.clear();

    try {
        // check which version it is...
        pugi::xml_node root = xmlDoc->document_element();

        ResponseDesc resp;
        StatusCode ret = ReadNetwork(root, &resp);
        if (ret != OK)
            description = resp.msg;
        return ret;
    } catch (const std::exception& e) {
        description = e.what();
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (...) {
        description = "Unknown exception thrown";
        parseSuccess = false;
        return UNEXPECTED;
    }

    return OK;
}

std::shared_ptr<IFormatParser> V2FormatParserCreator::create(size_t version) {
    return std::make_shared<FormatParser>(version);
}
