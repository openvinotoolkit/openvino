// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides interface for network reader that is used to build networks from a given IR
 * @file ie_icnn_net_reader.h
 */
#pragma once

#include <map>
#include <string>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_icnn_network.hpp"
#include "details/ie_no_copy.hpp"
#include "ie_api.h"

namespace InferenceEngine {
/**
 * @brief This class is the main interface to build and parse a network from a given IR
 *
 * All methods here do not throw exceptions and return a StatusCode and ResponseDesc object.
 * Alternatively, to use methods that throw exceptions, refer to the CNNNetReader wrapper class.
 */
class ICNNNetReader : public details::IRelease {
public:
    /**
     * @brief Parses the topology part of the IR (.xml)
     * This method can be called once only to read network. If you need to read another network instance then create new reader instance.
     * @param filepath The full path to the .xml file of the IR
     * @param resp Response message
     * @return Result code
     */
    virtual StatusCode ReadNetwork(const char *filepath, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Parses the topology part of the IR (.xml) given the xml as a buffer
     * This method can be called once only to read network. If you need to read another network instance then create new reader instance.
     * @param model Pointer to a char array with the IR
     * @param resp Response message
     * @param size Size of the char array in bytes
     * @return Result code
     */
    virtual StatusCode ReadNetwork(const void *model, size_t size, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Sets the weights buffer (.bin part) from the IR.
     * Weights Blob must always be of bytes - the casting to precision is done per-layer to support mixed
     * networks and to ease of use.
     * This method can be called more than once to reflect updates in the .bin.
     * @param weights Blob of bytes that holds all the IR binary data
     * @param resp Response message
     * @return Result code
    */
    virtual StatusCode SetWeights(const TBlob<uint8_t>::Ptr &weights, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Loads and sets the weights buffer directly from the IR .bin file.
     * This method can be called more than once to reflect updates in the .bin.
     * @param filepath Full path to the .bin file
     * @param resp Response message
     * @return Result code
     */
    virtual StatusCode ReadWeights(const char *filepath, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Returns a pointer to the built network
     * @param resp Response message
     */
    virtual ICNNNetwork *getNetwork(ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Retrieves the last building status
     * @param resp Response message
     */
    virtual bool isParseSuccess(ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Retrieves the last building failure message if failed
     * @param resp Response message
     * @return StatusCode that indicates the network status
     */
    virtual StatusCode getDescription(ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Gets network name
     * @param name Pointer to preallocated buffer that receives network name
     * @param len Length of the preallocated buffer, network name will be trimmed by this lenght
     * @param resp Response message
     * @return Result code
     */
    virtual StatusCode getName(char *name, size_t len, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Returns a version of IR
     * @param resp Response message
     * @return IR version number: 1 or 2
     */
    virtual int getVersion(ResponseDesc *resp) noexcept = 0;
};

/**
 * @brief Creates a CNNNetReader instance
 * @return An object that implements the ICNNNetReader interface
 */
INFERENCE_ENGINE_API(ICNNNetReader*)CreateCNNNetReader() noexcept;
}  // namespace InferenceEngine
