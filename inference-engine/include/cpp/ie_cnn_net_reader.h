// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Network reader class (wrapper) used to build networks from a given IR
 * 
 * @file ie_cnn_net_reader.h
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "details/ie_exception_conversion.hpp"
#include "details/os/os_filesystem.hpp"
#include "ie_blob.h"
#include "ie_cnn_network.h"
#include "ie_common.h"
#include "ie_icnn_net_reader.h"

namespace InferenceEngine {
/**
 * @deprecated Use InferenceEngine::Core::ReadNetwork methods. This API will be removed in 2020.3
 * @brief This is a wrapper class used to build and parse a network from the given IR.
 *
 * All the methods here can throw exceptions.
 */
IE_SUPPRESS_DEPRECATED_START
class INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::Core::ReadNetwork methods. This API will be removed in 2020.3")
    CNNNetReader {
public:
    /**
     * @brief A smart pointer to this class
     */
    using Ptr = std::shared_ptr<CNNNetReader>;

    /**
     * @brief A default constructor
     */
    CNNNetReader(): actual(InferenceEngine::CreateCNNNetReaderPtr()) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "CNNNetReader was not initialized.";
        }
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Resolve wstring path then call original ReadNetwork
     *
     * Wraps ICNNNetReader::ReadNetwork
     *
     * @param filepath The full path to the .xml file of the IR
     */
    void ReadNetwork(const std::wstring& filepath) {
        CALL_STATUS_FNC(ReadNetwork, details::wStringtoMBCSstringChar(filepath).c_str());
    }
#endif

    /**
     * @copybrief ICNNNetReader::ReadNetwork
     *
     * Wraps ICNNNetReader::ReadNetwork
     *
     * @param filepath The full path to the .xml file of the IR
     */
    void ReadNetwork(const std::string& filepath) {
        CALL_STATUS_FNC(ReadNetwork, filepath.c_str());
    }

    /**
     * @copybrief ICNNNetReader::ReadNetwork(const void*, size_t, ResponseDesc*)
     *
     * Wraps ICNNNetReader::ReadNetwork(const void*, size_t, ResponseDesc*)
     *
     * @param model Pointer to a char array with the IR
     * @param size Size of the char array in bytes
     */
    void ReadNetwork(const void* model, size_t size) {
        CALL_STATUS_FNC(ReadNetwork, model, size);
    }

    /**
     * @copybrief ICNNNetReader::SetWeights
     *
     * Wraps ICNNNetReader::SetWeights
     *
     * @param weights Blob of bytes that holds all the IR binary data
     */
    void SetWeights(const TBlob<uint8_t>::Ptr& weights) {
        CALL_STATUS_FNC(SetWeights, weights);
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Resolve wstring path then call original ReadWeights
     *
     * ICNNNetReader::ReadWeights
     *
     * @param filepath Full path to the .bin file
     */
    void ReadWeights(const std::wstring& filepath) {
        CALL_STATUS_FNC(ReadWeights, details::wStringtoMBCSstringChar(filepath).c_str());
    }
#endif

    /**
     * @copybrief ICNNNetReader::ReadWeights
     *
     * Wraps ICNNNetReader::ReadWeights
     *
     * @param filepath Full path to the .bin file
     */
    void ReadWeights(const std::string& filepath) {
        CALL_STATUS_FNC(ReadWeights, filepath.c_str());
    }

    /**
     * @brief Gets a copy of built network object
     *
     * @return A copy of the CNNNetwork object to be loaded
     */
    CNNNetwork getNetwork() {
        // network obj are to be updated upon this call
        if (network.get() == nullptr) {
            try {
                network.reset(new CNNNetwork(actual));
            } catch (...) {
                THROW_IE_EXCEPTION << "Could not allocate memory";
            }
        }
        return *network.get();
    }

    /**
     * @copybrief ICNNNetReader::isParseSuccess
     *
     * Wraps ICNNNetReader::isParseSuccess
     *
     * @return true if a parse is successful, false otherwise
     */
    bool isParseSuccess() const {
        CALL_FNC_NO_ARGS(isParseSuccess);
    }

    /**
     * @copybrief ICNNNetReader::getDescription
     *
     * Wraps ICNNNetReader::getDescription
     *
     * @return StatusCode that indicates the network status
     */
    std::string getDescription() const {
        CALL_STATUS_FNC_NO_ARGS(getDescription);
        return resp.msg;
    }

    /**
     * @copybrief ICNNNetReader::getName
     *
     * Wraps ICNNNetReader::getName
     *
     * @return String
     */
    std::string getName() const {
        char name[64];
        CALL_STATUS_FNC(getName, name, sizeof(name) / sizeof(*name));
        return name;
    }

    /**
     * @copybrief ICNNNetReader::getVersion
     *
     * Wraps ICNNNetReader::getVersion
     *
     * @return IR version number: 1 or 2
     */
    int getVersion() const {
        CALL_FNC_NO_ARGS(getVersion);
    }

private:
    CNNNetReaderPtr actual;
    std::shared_ptr<CNNNetwork> network;
};
IE_SUPPRESS_DEPRECATED_END
}  // namespace InferenceEngine
