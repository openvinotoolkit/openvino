// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for Inference Engine Extension Interface
 * @file ie_iextension.h
 */
#pragma once

#include "ie_api.h"
#include "ie_device.hpp"
#include "ie_layers.h"
#include "ie_error.hpp"
#include "ie_version.hpp"
#include <vector>
#include <string>
#include <memory>
#include <map>

#include "details/ie_no_copy.hpp"

/**
 * @def INFERENCE_EXTENSION_API(TYPE)
 * @brief Defines Inference Engine Extension API method
 */

#if defined(_WIN32) && defined(IMPLEMENT_INFERENCE_EXTENSION_API)
# define INFERENCE_EXTENSION_API(TYPE) extern "C"  __declspec(dllexport) TYPE
#else
# define INFERENCE_EXTENSION_API(TYPE) INFERENCE_ENGINE_API(TYPE)
#endif


namespace InferenceEngine {

/**
 * @struct DataConfig
 * @brief This structure describes data configuration
 */
struct DataConfig {
    /**
     * @brief Format of memory descriptor
     */
    TensorDesc desc;
    /**
     * @brief Index of in-place memory. If -1 memory cannot be in-place
     */
    int inPlace = -1;
    /**
     * @brief Flag for determination of the constant memory. If layer contains all constant memory we can calculate it on the load stage.
     */
    bool constant = false;
};

/**
 * @struct LayerConfig
 * @brief This structure describes Layer configuration
 */
struct LayerConfig {
    /**
     * @brief Supported dynamic batch. If false, dynamic batch is not supported
     */
    bool dynBatchSupport = false;
    /**
     * @brief Vector of input data configs
     */
    std::vector<DataConfig> inConfs;
    /**
     * @brief Vector of output data configs
     */
    std::vector<DataConfig> outConfs;
};

/**
 * @brief This class provides interface for extension implementations
 */
class ILayerImpl {
public:
    using Ptr = std::shared_ptr<ILayerImpl>;

    /**
     * @brief Destructor
     */
    virtual ~ILayerImpl() = default;

    /**
     * @brief Gets all supported configurations for the current layer
     * @param conf Vector with supported configurations
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Initializes the implementation
     * @param config Selected supported configuration
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode init(LayerConfig& config, ResponseDesc* resp) noexcept = 0;
};

/**
 * @brief This class provides interface for the implementation with the custom execution code
 */
class ILayerExecImpl : public ILayerImpl {
public:
    /**
     * @brief Execute method
     * @param inputs Vector of blobs with input memory
     * @param outputs Vector of blobs with output memory
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode execute(std::vector<Blob::Ptr>& inputs,
                               std::vector<Blob::Ptr>& outputs, ResponseDesc* resp) noexcept = 0;
};

/**
 * @brief This class provides interface for extension factories
 */
class ILayerImplFactory {
public:
    using Ptr = std::shared_ptr<ILayerImplFactory>;
    using ImplCreator = std::function<ILayerImpl*()>;

    /**
     * @brief Destructor
     */
    virtual ~ILayerImplFactory() = default;

    /**
     * @deprecated Implement IShapeInferImpl extension for shape inference.
     * @brief Sets output shapes by input shapes.
     * @param inShapes Shapes of all inputs coming in this layer
     * @param outShapes Generated shapes coming from this layer given the input
     * @param resp Response descriptor
     * @return Status code
     */
    INFERENCE_ENGINE_DEPRECATED
    virtual StatusCode getShapes(const std::vector<TensorDesc>& /*inShapes*/, std::vector<TensorDesc>& /*outShapes*/,
                                 ResponseDesc* /*resp*/) noexcept {
        return NOT_IMPLEMENTED;
    }

    /**
     * @brief Gets all possible implementations for the given cnn Layer
     * @param impls the vector with implementations which is ordered by priority
     * @param resp response descriptor
     * @return status code
     */
    virtual StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc* resp) noexcept = 0;
};

/**
 * @class IShapeInferImpl
 * @brief This class provides interface for the implementation with the custom execution code
 */
class IShapeInferImpl {
public:
    using Ptr = std::shared_ptr<IShapeInferImpl>;

    virtual ~IShapeInferImpl() = default;

    /**
     * @brief check that reshape can be applied, that parameters and shapes are valid
     */
    virtual StatusCode inferShapes(const std::vector<Blob::CPtr>& /*inBlobs*/,
                                   const std::map<std::string, std::string>& /*params*/,
                                   const std::map<std::string, Blob::Ptr>& /*blobs*/,
                                   std::vector<SizeVector>& /*outShapes*/,
                                   ResponseDesc* /*resp*/) noexcept { return NOT_IMPLEMENTED; }  // For backward-compatibility

    /**
     * @deprecated Use IShapeInferImpl::inferShapes(const std::vector<Blob::CPtr>&, const std::map<std::string, std::string>&,
                                   const std::map<std::string, Blob::Ptr>&, std::vector<SizeVector>&, ResponseDesc* ) noexcept.
     * @brief check that reshape can be applied, that parameters and shapes are valid
     */
    INFERENCE_ENGINE_DEPRECATED
    virtual StatusCode inferShapes(const std::vector<SizeVector>& /*inShapes*/,
                                   const std::map<std::string, std::string>& /*params*/,
                                   const std::map<std::string, Blob::Ptr>& /*blobs*/,
                                   std::vector<SizeVector>& /*outShapes*/,
                                   ResponseDesc* /*resp*/) noexcept {
        return NOT_IMPLEMENTED;
    }
};

/**
 * @class IShapeInferExtension
 * @brief This class is the reader extension interface to provide implementation for shape propagation
 */
class IShapeInferExtension : public InferenceEngine::details::IRelease {
public:
    /**
     * @brief Sets logging callback.
     * Logging is used to track what is going on inside.
     * @param listener Logging sink
     */
    virtual void SetLogCallback(InferenceEngine::IErrorListener& listener) noexcept = 0;

    /**
     * @brief Gets extension version information and stores in versionInfo
     * @param versionInfo Pointer to version info, will be set by plugin
     */
    virtual void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept = 0;

    /**
     * @brief Cleans resources up
     */
    virtual void Unload() noexcept = 0;

    /**
     * @brief Fills passed array with types of layers which shape infer implementations are included in the extension
     * @param types Array to store the layer types
     * @param size Size of the layer types array
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Gets shape propagation implementation for the given string-type of cnn Layer
     * @param impl the vector with implementations which is ordered by priority
     * @param resp response descriptor
     * @return status code
     */
    virtual StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl,
                                         const char* type,
                                         ResponseDesc* resp) noexcept = 0;
};

/**
 * @brief This class is the main extension interface
 */
class IExtension : public IShapeInferExtension {
public:
    virtual StatusCode getFactoryFor(ILayerImplFactory*& factory, const CNNLayer* cnnLayer,
                                     ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Fills passed array with types of layers which kernel implementations are included in the extension
     * @param types Array to store the layer types
     * @param size Size of the layer types array
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept = 0;

    StatusCode getShapeInferTypes(char**&, unsigned int&, ResponseDesc*) noexcept override {
        return NOT_IMPLEMENTED;
    };

    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr&, const char*, ResponseDesc*) noexcept override {
        return NOT_IMPLEMENTED;
    };
};

using IExtensionPtr = std::shared_ptr<IExtension>;
using IShapeInferExtensionPtr = std::shared_ptr<IShapeInferExtension>;

/**
 * @brief Creates the default instance of the extension
 * @param ext Extension interface
 * @param resp Response description
 * @return Status code
 */
INFERENCE_EXTENSION_API(StatusCode) CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept;

/**
 * @brief Creates the default instance of the shape infer extension
 * @param ext Shape Infer Extension interface
 * @param resp Response description
 * @return Status code
 */
INFERENCE_EXTENSION_API(StatusCode) CreateShapeInferExtension(IShapeInferExtension*& ext, ResponseDesc* resp) noexcept;


}  // namespace InferenceEngine
