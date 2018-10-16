// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the main MKL-DNN Extension API
 * @file mkldnn_extension.hpp
 */
#pragma once

#include <ie_iextension.h>

#include "mkldnn_generic_primitive.hpp"

namespace InferenceEngine {
namespace MKLDNNPlugin {

/**
 * @deprecated use new extensibility API
 * @brief The IMKLDNNExtension class provides the main extension interface
 */
class IMKLDNNExtension : public IExtension {
public:
    /**
     * @brief Creates a generic layer and returns a pointer to an instance
     * @param primitive Pointer to newly created layer
     * @param layer Layer parameters (source for name, type, precision, attr, weights...)
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success
     */
    virtual InferenceEngine::StatusCode CreateGenericPrimitive(IMKLDNNGenericPrimitive*& primitive,
                                          const InferenceEngine::CNNLayerPtr& layer,
                                          InferenceEngine::ResponseDesc *resp) const noexcept = 0;
    /**
     * @brief This method isn't implemented for the old API
     */
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        return NOT_IMPLEMENTED;
    };
    /**
     * @brief This method isn't implemented for the old API
     */
    StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp) noexcept override {
        return NOT_IMPLEMENTED;
    }

    /**
     * @brief Gets shape propagation implementation for the given string-type of cnn Layer
     * @param impl the vector with implementations which is ordered by priority
     * @param resp response descriptor
     * @return status code
     */
    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept override {
        return NOT_IMPLEMENTED;
    };
};

/**
 * @deprecated use new extensibility API
 * @brief Creates the default instance of the extension
 * @return The MKL-DNN Extension interface
 */
INFERENCE_EXTENSION_API(StatusCode) CreateMKLDNNExtension(IMKLDNNExtension*& ext, ResponseDesc* resp) noexcept;

}  // namespace MKLDNNPlugin
}  // namespace InferenceEngine
