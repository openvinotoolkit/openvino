// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that defines a wrapper class for handling extension instantiation and releasing resources
 * @file ie_extension.h
 */
#pragma once

#include "details/ie_so_pointer.hpp"
#include "ie_iextension.h"
#include <string>
#include <memory>
#include <map>

namespace InferenceEngine {
namespace details {

/**
 * @brief The SOCreatorTrait class specialization for IExtension case, defines the name of the fabric method for creating IExtension object in DLL
 */
template<>
class SOCreatorTrait<IExtension> {
public:
    /**
     * @brief A name of the fabric method for creating an IExtension object in DLL
     */
    static constexpr auto name = "CreateExtension";
};

/**
 * @brief The SOCreatorTrait class specialization for IExtension case, defines the name of the fabric method for creating IExtension object in DLL
 */
template<>
class SOCreatorTrait<IShapeInferExtension> {
public:
    /**
     * @brief A name of the fabric method for creating an IShapeInferExtension object in DLL
     */
    static constexpr auto name = "CreateShapeInferExtension";
};

}  // namespace details

/**
 * @brief This class is a C++ helper to work with objects created using extensions.
 */
class Extension : public IExtension {
public:
    /**
   * @brief Loads extension from a shared library
   * @param name Full or relative path to extension library
   */
    explicit Extension(const file_name_t &name)
            : actual(name) {}

    /**
     * @brief Gets the extension version information
     * @param versionInfo A pointer to version info, set by the plugin
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
     * @brief Does nothing since destruction is done via the regular mechanism
     */
    void Release() noexcept override {}

    /**
     * @brief Gets the array with types of layers which are included in the extension
     * @param types Types array
     * @param size Size of the types array
     * @param resp Response descriptor
     * @return Status code
     */
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        return actual->getPrimitiveTypes(types, size, resp);
    }

    /**
     * @brief Gets the factory with implementations for a given layer
     * @param factory Factory with implementations
     * @param cnnLayer A layer to get the factory for
     * @param resp Response descriptor
     * @return Status code
     */
    StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer,
                                     ResponseDesc *resp) noexcept override {
        return actual->getFactoryFor(factory, cnnLayer, resp);
    }

    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept override {
        return actual->getShapeInferImpl(impl, type, resp);
    }

protected:
    /**
    * @brief A SOPointer instance to the loaded templated object
    */
    InferenceEngine::details::SOPointer<IExtension> actual;
};

/**
 * @brief This class is a C++ helper to work with objects created using extensions.
 */
class ShapeInferExtension : public IShapeInferExtension {
public:
    /**
   * @brief Loads extension from a shared library
   * @param name Full or relative path to extension library
   */
    explicit ShapeInferExtension(const file_name_t &name)
            : actual(name) {}

    /**
     * @brief Gets the extension version information
     * @param versionInfo A pointer to version info, set by the plugin
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
     * @brief Does nothing since destruction is done via the regular mechanism
     */
    void Release() noexcept override {}

    /**
     * @brief Gets the array with types of layers which are included in the extension
     * @param types Types array
     * @param size Size of the types array
     * @param resp Response descriptor
     * @return Status code
     */
    StatusCode getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        return actual->getShapeInferTypes(types, size, resp);
    }

    /**
     * @brief Gets shape propagation implementation for the given string-type of cnn Layer
     * @param impl the vector with implementations which is ordered by priority
     * @param resp response descriptor
     * @return status code
     */
    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept override {
        return actual->getShapeInferImpl(impl, type, resp);
    }

protected:
    /**
    * @brief A SOPointer instance to the loaded templated object
    */
    InferenceEngine::details::SOPointer<IShapeInferExtension> actual;
};

/**
 * @brief Creates a special shared_pointer wrapper for the given type from a specific shared module
 * @param name Name of the shared library file
 * @return shared_pointer A wrapper for the given type from a specific shared module
 */
template<>
inline std::shared_ptr<IShapeInferExtension> make_so_pointer(const file_name_t &name) {
    return std::make_shared<ShapeInferExtension>(name);
}

/**
 * @brief Creates a special shared_pointer wrapper for the given type from a specific shared module
 * @param name Name of the shared library file
 * @return shared_pointer A wrapper for the given type from a specific shared module
 */
template<>
inline std::shared_ptr<IExtension> make_so_pointer(const file_name_t &name) {
    return std::make_shared<Extension>(name);
}

}  // namespace InferenceEngine
