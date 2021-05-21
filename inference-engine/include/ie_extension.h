// Copyright (C) 2018-2021 Intel Corporation
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

#include <ngraph/ngraph.hpp>
#include "ie_iextension.h"
#include "details/ie_so_pointer.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief The SOCreatorTrait class specialization for IExtension case, defines the name of the fabric method for
 * creating IExtension object in DLL
 */
template <>
class SOCreatorTrait<IExtension> {
public:
    /**
     * @brief A name of the fabric method for creating an IExtension object in DLL
     */
    static constexpr auto name = "CreateExtension";
};

}  // namespace details

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
    template <typename C,
              typename = details::enableIfSupportedChar<C>>
    explicit Extension(const std::basic_string<C>& name): actual(name) {}

    /**
     * @brief Gets the extension version information
     *
     * @param versionInfo A pointer to version info, set by the plugin
     */
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        actual->GetVersion(versionInfo);
    }

    /**
     * @brief Cleans the resources up
     */
    void Unload() noexcept override {
        actual->Unload();
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
        if (node == nullptr) IE_THROW() << "Provided ngraph::Node pointer is nullptr.";
        return actual->getImplTypes(node);
    }

    /**
     * @brief Returns implementation for specific nGraph op
     * @param node shared pointer to nGraph op
     * @param implType implementation type
     * @return shared pointer to implementation
     */
    ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
        if (node == nullptr) IE_THROW() << "Provided ngraph::Node pointer is nullptr.";
        return actual->getImplementation(node, implType);
    }

protected:
    /**
     * @brief A SOPointer instance to the loaded templated object
     */
    details::SOPointer<IExtension> actual;
};

/**
 * @brief Creates extension using deprecated API
 * @tparam T extension type
 * @param name extension library name
 * @return shared pointer to extension
 */
template<typename T = IExtension>
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
template<typename T = IExtension>
INFERENCE_ENGINE_DEPRECATED("Use std::make_shared<Extension>")
inline std::shared_ptr<IExtension> make_so_pointer(const std::wstring& name) {
    return std::make_shared<Extension>(name);
}

#endif
}  // namespace InferenceEngine
