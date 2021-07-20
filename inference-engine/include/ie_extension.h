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

// ===========================================================================================================================================
// New extensions experiment
// ===========================================================================================================================================
class INFERENCE_ENGINE_API_CLASS(NewExtension) {
public:
    using Ptr = std::shared_ptr<NewExtension>;

    using type_info_t = ngraph::Node::type_info_t;

    virtual ~NewExtension() = default;

    virtual const type_info_t& get_type_info() const = 0;
};

class INFERENCE_ENGINE_API_CLASS(OpsetExtension): public NewExtension {
public:
    virtual std::map<std::string, ngraph::OpSet> getOpSets() = 0;

    NGRAPH_RTTI_DECLARATION;
};

class INFERENCE_ENGINE_API_CLASS(ExtensionContainer) {
public:
    using Ptr = std::shared_ptr<ExtensionContainer>;
    virtual ~ExtensionContainer() = default;
    virtual const std::vector<NewExtension::Ptr>& getExtensions() const = 0;
    operator const std::vector<NewExtension::Ptr>() const {
        return getExtensions();
    }
};

class INFERENCE_ENGINE_API_CLASS(DefaultExtensionContainer) final : public ExtensionContainer {
public:
    using Ptr = std::shared_ptr<ExtensionContainer>;
    DefaultExtensionContainer(const std::vector<NewExtension::Ptr>& extensions): extensions(extensions) {} // NOLINT

    const std::vector<NewExtension::Ptr>& getExtensions() const override {
        return extensions;
    }

    void addExtension(const NewExtension::Ptr& extension) {
        extensions.emplace_back(extension);
    }

    void addExtension(const std::vector<NewExtension::Ptr>& extensions) {
        for (const auto& extension : extensions)
            this->extensions.emplace_back(extension);
    }

private:
    std::vector<NewExtension::Ptr> extensions;
};

namespace details {

/**
 * @brief The SOCreatorTrait class specialization for IExtension case, defines the name of the fabric method for
 * creating IExtension object in DLL
 */
template <>
class SOCreatorTrait<ExtensionContainer> {
public:
    /**
     * @brief A name of the fabric method for creating an IExtension object in DLL
     */
    static constexpr auto name = "CreateExtensionContainer";
};

}  // namespace details

class INFERENCE_ENGINE_API_CLASS(SOExtension) final: public NewExtension {
public:
    SOExtension(const details::SharedObjectLoader& actual, const NewExtension::Ptr& ext): actual(actual), extension(ext) {}
    const NewExtension::Ptr& getExtension() {
        return extension;
    }
    const type_info_t& get_type_info() const override {
        return extension->get_type_info();
    }

private:
    const details::SharedObjectLoader& actual;
    NewExtension::Ptr extension;
};

class INFERENCE_ENGINE_API_CLASS(SOExtensionContainer) final: public ExtensionContainer {
public:
    template <typename C,
              typename = details::enableIfSupportedChar<C>>
    explicit SOExtensionContainer(const std::basic_string<C>& name): actual(name) {}

    const std::vector<NewExtension::Ptr>& getExtensions() const override {
        const auto& ext = actual->getExtensions();
        static std::vector<NewExtension::Ptr> extensions;
        if (extensions.empty()) {
            for (const auto& ex : ext) {
                extensions.emplace_back(std::make_shared<SOExtension>(actual, ex));
            }
        }
        return extensions;
    }

private:
    details::SOPointer<ExtensionContainer> actual;
};

/**
 * @brief Creates the default instance of the extension
 *
 * @param ext Extension interface
 */
INFERENCE_EXTENSION_API(void) CreateExtensionContainer(ExtensionContainer::Ptr& ext);
INFERENCE_EXTENSION_API(void) CreateExtensions(std::vector<InferenceEngine::NewExtension::Ptr>&);

/**
 * @def IE_DEFINE_EXTENSION_CREATE_FUNCTION
 * @brief Generates extension creation function
 */
#define IE_CREATE_CONTAINER(ContainerType)                                                                                              \
INFERENCE_EXTENSION_API(void) InferenceEngine::CreateExtensionContainer(std::shared_ptr<InferenceEngine::ExtensionContainer>& ext) {    \
    ext = std::make_shared<ContainerType>();                                                                                            \
}

#define IE_CREATE_DEFAULT_CONTAINER(extensions)                                                                                         \
INFERENCE_EXTENSION_API(void) InferenceEngine::CreateExtensionContainer(std::shared_ptr<InferenceEngine::ExtensionContainer>& ext) {    \
    ext = std::make_shared<DefaultExtensionContainer>(extensions);                                                                      \
}

#define IE_CREATE_EXTENSIONS(extensions)                                                                                                \
INFERENCE_EXTENSION_API(void) InferenceEngine::CreateExtensions(std::vector<InferenceEngine::NewExtension::Ptr>& ext) {                 \
    ext = extensions;                                                                                                                   \
}

template <typename C,
         typename = details::enableIfSupportedChar<C>>
std::vector<NewExtension::Ptr> load_extensions(const std::basic_string<C>& name) {
    details::SharedObjectLoader so(name.c_str());
    std::vector<NewExtension::Ptr> extensions;
    try {
        using CreateF = void(std::vector<InferenceEngine::NewExtension::Ptr>&);
        std::vector<InferenceEngine::NewExtension::Ptr> ext;
        reinterpret_cast<CreateF*>(so.get_symbol("CreateExtensions"))(ext);
        for (const auto& ex : ext) {
            extensions.emplace_back(std::make_shared<SOExtension>(so, ex));
        }
    } catch(...) {details::Rethrow();}
    return extensions;
}
}  // namespace InferenceEngine
