// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/ie_irelease.hpp>
#include <cpp/ie_cnn_network.h>
#include <ie_iextension.h>
#include <istream>
#include <string>
#include <vector>
#include <ie_api.h>

namespace InferenceEngine {

/**
 * @brief IReader an abstract interface for Inference Engine readers
 */
class INFERENCE_ENGINE_API_CLASS(IReader): public details::IRelease {
public:
    /**
     * @brief Checks that reader supports format of the model
     * @param model stream with model
     * @return true if format is supported
     */
    virtual bool supportModel(std::istream& model) const = 0;
    /**
     * @brief Reads the model to CNNNetwork
     * @param model stream with model
     * @param exts vector with extensions
     *
     * @return CNNNetwork
     */
    virtual CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const = 0;
    /**
     * @brief Reads the model to CNNNetwork
     * @param model stream with model
     * @param weights blob with binary data
     * @param exts vector with extensions
     *
     * @return CNNNetwork
     */
    virtual CNNNetwork read(std::istream& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) const = 0;

    /**
     * @brief Returns all supported extensions for data files
     *
     * @return vector of file extensions, for example the reader for OpenVINO IR returns {"bin"}
     */
    virtual std::vector<std::string> getDataFileExtensions() const = 0;
};

/**
 * @brief Creates the default instance of the reader
 *
 * @param reader Reader interface
 * @param resp Response description
 * @return Status code
 */
INFERENCE_PLUGIN_API(StatusCode) CreateReader(IReader*& reader, ResponseDesc* resp) noexcept;

}  // namespace InferenceEngine
