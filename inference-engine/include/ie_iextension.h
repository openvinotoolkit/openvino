// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for Inference Engine Extension Interface
 *
 * @file ie_iextension.h
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "details/ie_no_copy.hpp"
#include "ie_api.h"
#include "ie_error.hpp"
#include "ie_common.h"
#include "ie_layouts.h"
#include "ie_blob.h"
#include "ie_layers.h"
#include "ie_version.hpp"

/**
 * @def INFERENCE_EXTENSION_API(TYPE)
 * @brief Defines Inference Engine Extension API method
 */

#if defined(_WIN32) && defined(IMPLEMENT_INFERENCE_EXTENSION_API)
#define INFERENCE_EXTENSION_API(TYPE) extern "C" __declspec(dllexport) TYPE
#else
#define INFERENCE_EXTENSION_API(TYPE) INFERENCE_ENGINE_API(TYPE)
#endif

namespace ngraph {

class OpSet;
class Node;

}  // namespace ngraph

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
     * @brief Flag for determination of the constant memory. If layer contains all constant memory we can calculate it
     * on the load stage.
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
 * @interface ILayerImpl
 * @brief This class provides interface for extension implementations
 */
class INFERENCE_ENGINE_API_CLASS(ILayerImpl) {
public:
    /**
     * @brief A shared pointer to the ILayerImpl interface
     */
    using Ptr = std::shared_ptr<ILayerImpl>;

    /**
     * @brief Destructor
     */
    virtual ~ILayerImpl();
};

/**
 * @interface ILayerExecImpl
 * @brief This class provides interface for the implementation with the custom execution code
 */
class INFERENCE_ENGINE_API_CLASS(ILayerExecImpl) : public ILayerImpl {
public:
    /**
     * @brief A shared pointer to the ILayerExecImpl interface
     */
    using Ptr = std::shared_ptr<ILayerExecImpl>;

    /**
     * @brief Destructor
     */
    virtual ~ILayerExecImpl();

    /**
     * @brief Gets all supported configurations for the current layer
     *
     * @param conf Vector with supported configurations
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Initializes the implementation
     *
     * @param config Selected supported configuration
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode init(LayerConfig& config, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Execute method
     *
     * @param inputs Vector of blobs with input memory
     * @param outputs Vector of blobs with output memory
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                               ResponseDesc* resp) noexcept = 0;
};

/**
 * @deprecated Implement IExtension::getImplTypes and IExtension::getImplementation
 * The interface will be removed in 2021.1 release.
 * @interface ILayerImplFactory
 * @brief This class provides interface for extension factories
 */
class INFERENCE_ENGINE_DEPRECATED("Implement IExtension::getImplTypes and IExtension::getImplementation") ILayerImplFactory {
public:
    /**
     * @brief A shared pointer to the ILayerImplFactory interface
     */
    IE_SUPPRESS_DEPRECATED_START
    using Ptr = std::shared_ptr<ILayerImplFactory>;
    IE_SUPPRESS_DEPRECATED_END

    using ImplCreator = std::function<ILayerImpl*()>;

    /**
     * @brief Destructor
     */
    virtual ~ILayerImplFactory() = default;

    /**
     * @brief Gets all possible implementations for the given cnn Layer
     *
     * @param impls the vector with implementations which is ordered by priority
     * @param resp response descriptor
     * @return status code
     */
    virtual StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc* resp) noexcept = 0;
};

/**
 * @brief This class is the main extension interface
 */
class INFERENCE_ENGINE_API_CLASS(IExtension) : public InferenceEngine::details::IRelease {
public:
    /**
     * @deprecated Use IExtension::getImplementation to get a concrete implementation
     * The method will be removed in 2021.1 release.
     * @brief Provides a factory for a specified CNNLayer
     * @param factory A factory returned from an extension plugin
     * @param cnnLayer A CNNLayer object to provide factory for
     * @param resp Response descriptor
     * @return Status code
     */
    IE_SUPPRESS_DEPRECATED_START
    INFERENCE_ENGINE_DEPRECATED("Use IExtension::getImplementation to get a concrete implementation")
    virtual StatusCode getFactoryFor(ILayerImplFactory*& factory, const CNNLayer* cnnLayer,
                                     ResponseDesc* resp) noexcept {
        (void)factory;
        (void)cnnLayer;
        (void)resp;
        return NOT_IMPLEMENTED;
    }
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @deprecated Use IExtension::getImplTypes to get implementation types for a particular node
     * The method will be removed in 2021.1 release.
     * @brief Fills passed array with types of layers which kernel implementations are included in the extension
     *
     * @param types Array to store the layer types
     * @param size Size of the layer types array
     * @param resp Response descriptor
     * @return Status code
     */
    INFERENCE_ENGINE_DEPRECATED("Use IExtension::getImplTypes to get implementation types for a particular node")
    virtual StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
        (void)types;
        (void)size;
        (void)resp;
        return NOT_IMPLEMENTED;
    }

    /**
     * @brief Returns operation sets
     * This method throws an exception if it was not implemented
     * @return map of opset name to opset
     */
    virtual std::map<std::string, ngraph::OpSet> getOpSets();

    /**
     * @brief Returns vector of implementation types
     * @param node shared pointer to nGraph op
     * @return vector of strings
     */
    virtual std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) {
        (void)node;
        return {};
    }

    /**
     * @brief Returns implementation for specific nGraph op
     * @param node shared pointer to nGraph op
     * @param implType implementation type
     * @return shared pointer to implementation
     */
    virtual ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
        (void)node;
        (void)implType;
        return nullptr;
    }

    /**
     * @brief Cleans resources up
     */
    virtual void Unload() noexcept = 0;

    /**
     * @brief Gets extension version information and stores in versionInfo
     * @param versionInfo Pointer to version info, will be set by plugin
     */
    virtual void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept = 0;
};

/**
 * @brief A shared pointer to a IExtension interface
 */
using IExtensionPtr = std::shared_ptr<IExtension>;

/**
 * @brief Creates the default instance of the extension
 *
 * @param ext Extension interface
 * @param resp Response description
 * @return Status code
 */
INFERENCE_EXTENSION_API(StatusCode) CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept;

}  // namespace InferenceEngine
