// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <ie_cnn_net_reader_impl.h>

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cnn_network_ngraph_impl.hpp"
#include "details/os/os_filesystem.hpp"
#include "ie_format_parser.h"
#include "ie_ir_reader.hpp"
#include "ie_profiling.hpp"
#include "ie_plugin.hpp"
#include "parsers.h"
#include "xml_parse_utils.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

IE_SUPPRESS_DEPRECATED_START
CNNNetReaderImpl::CNNNetReaderImpl(const FormatParserCreator::Ptr& _creator)
    : parseSuccess(false), _version(0), parserCreator(_creator) {}

CNNNetReaderImpl::~CNNNetReaderImpl() { }

StatusCode CNNNetReaderImpl::SetWeights(const TBlob<uint8_t>::Ptr& weights, ResponseDesc* desc) noexcept {
    if (!_parser && _version < 10) {
        return DescriptionBuffer(desc) << "network must be read first";
    }
    try {
        if (_version == 10) {
            // It's time to perform actual reading of V10 network and instantiate CNNNetworkNGraphImpl
            IRReader v10Reader(extensions);
            std::stringstream model;
            xmlDoc->save(model);
            network = std::make_shared<CNNNetworkNGraphImpl>(v10Reader.read(model.str(), weights));
        } else {
            _parser->SetWeights(weights);
        }
    } catch (const InferenceEngineException& iee) {
        xmlDoc.reset();
        return DescriptionBuffer(desc) << iee.what();
    }

    xmlDoc.reset();
    return OK;
}

size_t CNNNetReaderImpl::GetFileVersion(pugi::xml_node& root) {
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

StatusCode CNNNetReaderImpl::ReadWeights(const char* filepath, ResponseDesc* resp) noexcept {
    IE_PROFILING_AUTO_SCOPE(CNNNetReaderImpl::ReadWeights)
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
        FileUtils::readAllFile(filepath, weightsPtr->buffer(), ulFileSize);
        return SetWeights(weightsPtr, resp);
    } catch (const InferenceEngineException& ex) {
        return DescriptionBuffer(resp) << ex.what();
    }
}

StatusCode CNNNetReaderImpl::ReadNetwork(const char* filepath, ResponseDesc* resp) noexcept {
    IE_PROFILING_AUTO_SCOPE(CNNNetReaderImpl::ReadNetwork)
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

StatusCode CNNNetReaderImpl::ReadNetwork() {
    description.clear();

    try {
        // check which version it is...
        pugi::xml_node root = xmlDoc->document_element();

        _version = GetFileVersion(root);
        if (_version < 2) THROW_IE_EXCEPTION << "deprecated IR version: " << _version;
        if (_version == 10) {
            // Activate an alternative code path for V10 that should be read into ngraph::Function
            // We cannot proceed with reading right now, because there is not binary file loaded.
            // So we are postponing real read until weights are specified.
            parseSuccess = true;
        } else if (_version < 10) {
            _parser = parserCreator->create(_version);
            InferenceEngine::details::CNNNetworkImplPtr local_network = _parser->Parse(root);
            name = local_network->getName();
            local_network->validate(_version);
            network = local_network;
            parseSuccess = true;
        } else {
            THROW_IE_EXCEPTION << "cannot parse future versions: " << _version;
        }
    } catch (const std::string& err) {
        description = err;
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (const InferenceEngineException& e) {
        description = e.what();
        parseSuccess = false;
        return GENERAL_ERROR;
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

void CNNNetReaderImpl::addExtensions(const std::vector<InferenceEngine::IExtensionPtr>& ext) {
    extensions = ext;
}

std::shared_ptr<IFormatParser> V2FormatParserCreator::create(size_t version) {
    return std::make_shared<FormatParser>(version);
}

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateICNNNetReader(ICNNNetReader *& data, ResponseDesc *resp) noexcept {
    data = new CNNNetReaderImpl(std::make_shared<V2FormatParserCreator>());
    return StatusCode::OK;
}

IE_SUPPRESS_DEPRECATED_END
