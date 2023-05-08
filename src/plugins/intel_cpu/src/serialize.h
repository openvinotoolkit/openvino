// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "extension_mngr.h"

#include <iostream>
#include <functional>
#include <cpp/ie_cnn_network.h>
#include "ngraph/runtime/aligned_buffer.hpp"

namespace ov {
namespace intel_cpu {

class CNNNetworkSerializer {
public:
    CNNNetworkSerializer(std::ostream & ostream, ExtensionManager::Ptr extensionManager);
    void operator << (const InferenceEngine::CNNNetwork & network);

private:
    std::ostream & _ostream;
    ExtensionManager::Ptr _extensionManager;
};

class CNNNetworkDeserializer {
    class BufferParser {
    public:
        BufferParser(void* buffer, size_t size) : _buffer(buffer), _size(size), _offset(0) {}
        void read(char* target, size_t size) {
            if (target && _buffer && _offset + size <= _size) {
                std::memcpy(target, static_cast<char *>(_buffer) + _offset, size);
                _offset += size;
            } else {
                IE_THROW(NetworkNotRead) << "The inputs model information is invalid.";
            }
        }
        void seekg(size_t offset) {
            _offset = offset;
        }
        void skip(size_t size) {
            _offset += size;
        }
        void* get_ptr() {
            return static_cast<char *>(_buffer);
        }

    private:
        void* _buffer;
        size_t _size;
        size_t _offset;
    };

public:
    typedef std::function<InferenceEngine::CNNNetwork(const std::string&, const InferenceEngine::Blob::CPtr&)>
        cnn_network_builder;
    CNNNetworkDeserializer(std::istream& istream, cnn_network_builder fn);
    CNNNetworkDeserializer(std::shared_ptr<ngraph::runtime::AlignedBuffer>& _buffer, cnn_network_builder fn);
    void operator>>(InferenceEngine::CNNNetwork& network);
    void parse_buffer(InferenceEngine::CNNNetwork & network);
    void parse_stream(InferenceEngine::CNNNetwork & network);

private:
    std::istream* _istream;
    std::shared_ptr<ngraph::runtime::AlignedBuffer> _network_buffer;
    cnn_network_builder _cnn_network_builder;
};

// const std::string& model, const Blob::CPtr& weights

}   // namespace intel_cpu
}   // namespace ov
