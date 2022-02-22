// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that defines a wrapper class for handling extension instantiation and releasing resources
 *
 * @file ie_extension.h
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_iextension.h"
#include "ngraph/opsets/opset.hpp"

namespace InferenceEngine {

/**
 * @brief This class is a C++ helper to work with objects created using extensions.
 */
class INFERENCE_ENGINE_API_CLASS(Extension) final : public IExtension {
public:
    /**
     * @brief Loads extension from a shared library
     *
     * @param name Full or relative path to extension library
     */
    explicit Extension(const std::string& name);

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Loads extension from a shared library
     *
     * @param name Full or relative path to extension library
     */
    explicit Extension(const std::wstring& name);
#endif  // ENABLE_UNICODE_PATH_SUPPORT

    /**
     * @brief Gets the extension version information
     *
     * @param versionInfo A pointer to version info, set by the plugin
     */
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        _actual->GetVersion(versionInfo);
    }

    /**
     * @brief Cleans the resources up
     */
    void Unload() noexcept override {
        _actual->Unload();
    }

    /**
     * @brief Returns operation sets
     * This method throws an exception if it was not implemented
     * @return map of opset name to opset
     */
    std::map<std::string, ngraph::OpSet> getOpSets() override;

    /**
     * @brief Returns vector of implementation types
     * @param node shared pointer to nGraph op
     * @return vector of strings
     */
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
        if (node == nullptr)
            IE_THROW() << "Provided ngraph::Node pointer is nullptr.";
        return _actual->getImplTypes(node);
    }

    /**
     * @brief Returns implementation for specific nGraph op
     * @param node shared pointer to nGraph op
     * @param implType implementation type
     * @return shared pointer to implementation
     */
    ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
        if (node == nullptr)
            IE_THROW() << "Provided ngraph::Node pointer is nullptr.";
        return _actual->getImplementation(node, implType);
    }

protected:
    /**
     * @brief A shared library
     */
    std::shared_ptr<void> _so;

    /**
     * @brief A instance to the loaded templated object
     */
    std::shared_ptr<InferenceEngine::IExtension> _actual;
};

/**
 * @brief Creates extension using deprecated API
 * @tparam T extension type
 * @param name extension library name
 * @return shared pointer to extension
 */
template <typename T = IExtension>
INFERENCE_ENGINE_DEPRECATED("Use std::make_shared<Extension>")
inline std::shared_ptr<T> make_so_pointer(const std::string& name) {
    return std::make_shared<Extension>(name);
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Creates extension using deprecated API
 * @param name extension library name
 * @return shared pointer to extension
 */
template <typename T = IExtension>
INFERENCE_ENGINE_DEPRECATED("Use std::make_shared<Extension>")
inline std::shared_ptr<IExtension> make_so_pointer(const std::wstring& name) {
    return std::make_shared<Extension>(name);
}

#endif
}  // namespace InferenceEngine
