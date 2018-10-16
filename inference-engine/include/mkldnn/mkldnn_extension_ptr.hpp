// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that defines a wrapper class for handling extension instantiation and releasing resources
 * @file mkldnn_extension_ptr.hpp
 */
#pragma once

#include "details/ie_so_pointer.hpp"
#include "mkldnn/mkldnn_extension.hpp"
#include <string>
#include <memory>

namespace InferenceEngine {
namespace details {

/**
 * @deprecated use new extensibility API
 * @brief The SOCreatorTrait class defines the name of the fabric
 * for creating MKLDNNPlugin::IMKLDNNExtension object in DLL
 */
template<>
class SOCreatorTrait<MKLDNNPlugin::IMKLDNNExtension> {
public:
    /**
     * @brief A name of the fabric for creating an MKLDNNPlugin::IMKLDNNExtension object in DLL
     */
    static constexpr auto name = "CreateMKLDNNExtension";
};

}  // namespace details

namespace MKLDNNPlugin {

/**
 * @deprecated use new extensibility API
 * @brief This class is a C++ helper to work with objects created using extensions.
 * Implements different interfaces.
 */
class MKLDNNExtension : public MKLDNNPlugin::IMKLDNNExtension {
public:
    /**
   * @brief Loads extension from a shared library
   * @param name Logical name of the extension library (soname without .dll/.so/lib prefix)
   */
    explicit MKLDNNExtension(const std::string &name)
            : actual(name) {}

    /**
     * @brief Creates a generic layer and returns a pointer to an instance
     * @param primitive Pointer to a newly created layer
     * @param layer Layer parameters (source for name, type, precision, attr, weights...)
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success
     */
    InferenceEngine::StatusCode CreateGenericPrimitive(IMKLDNNGenericPrimitive *&primitive,
                                                       const InferenceEngine::CNNLayerPtr &layer,
                                                       InferenceEngine::ResponseDesc *resp) const noexcept override {
        return actual->CreateGenericPrimitive(primitive, layer, resp);
    }

    /**
     * @brief This method isn't implemented for the old API
     */
    InferenceEngine::StatusCode getPrimitiveTypes(char**& types, unsigned int& size,
                                                  InferenceEngine::ResponseDesc* resp) noexcept override {
        return actual->getPrimitiveTypes(types, size, resp);
    }

    /**
     * @brief This method isn't implemented for the old API
     */
    InferenceEngine::StatusCode getFactoryFor(InferenceEngine::ILayerImplFactory *&factory,
                                              const InferenceEngine::CNNLayer *cnnLayer,
                                              InferenceEngine::ResponseDesc *resp) noexcept override {
        return actual->getFactoryFor(factory, cnnLayer, resp);
    }

    /**
     * @brief This method isn't implemented for the old API
     */
    InferenceEngine::StatusCode getShapeInferImpl(InferenceEngine::IShapeInferImpl::Ptr& impl, const char* type,
                                                  InferenceEngine::ResponseDesc* resp) noexcept override {
        return actual->getShapeInferImpl(impl, type, resp);
    };

    /**
     * @brief Gets the extension version information
     * @param versionInfo A pointer to version info, set by plugin
     */
    void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept override {
        actual->GetVersion(versionInfo);
    }

    /**
     * @brief Sets a log callback that is used to track what is going on inside
     * @param listener Logging listener
     */
    void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept override {
        actual->SetLogCallback(listener);
    }

    /**
     * @brief Cleans the resources up
     */
    void Unload() noexcept override {
        actual->Unload();
    }

    /**
     * @brief Does nothing since destruction is done via regular mechanism
     */
    void Release() noexcept override {}

protected:
    /**
    * @brief An SOPointer instance to the loaded templated object
    */
    InferenceEngine::details::SOPointer<MKLDNNPlugin::IMKLDNNExtension> actual;
};
}  // namespace MKLDNNPlugin


/**
 * @deprecated use new extensibility API
 * @brief Creates a special shared_pointer wrapper for the given type from a specific shared module
 * @param name Name of the shared library file
 * @return shared_pointer A wrapper for the given type from a specific shared module
 */
template<>
inline std::shared_ptr<MKLDNNPlugin::IMKLDNNExtension> make_so_pointer(const std::string &name) {
    return std::make_shared<MKLDNNPlugin::MKLDNNExtension>(name);
}

}  // namespace InferenceEngine
