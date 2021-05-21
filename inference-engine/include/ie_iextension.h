// Copyright (C) 2018-2021 Intel Corporation
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

#include "ie_api.h"
#include "ie_common.h"
#include "ie_layouts.h"
#include "ie_blob.h"
#include "ie_version.hpp"
#include <ngraph/ngraph.hpp>

/**
 * @def INFERENCE_EXTENSION_API(TYPE)
 * @brief Defines Inference Engine Extension API method
 */
#if defined(_WIN32) && defined(IMPLEMENT_INFERENCE_EXTENSION_API)
#define INFERENCE_EXTENSION_API(TYPE) extern "C" __declspec(dllexport) TYPE
#else
#define INFERENCE_EXTENSION_API(TYPE) INFERENCE_ENGINE_API(TYPE)
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
 * @brief This class is the main extension interface
 */
class INFERENCE_ENGINE_API_CLASS(IExtension) : public std::enable_shared_from_this<IExtension> {
public:
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

    /**
     * @brief Implements deprecated API
     */
    INFERENCE_ENGINE_DEPRECATED("Do not override or use this method. Use IE_DEFINE_EXTENSION_CREATE_FUNCTION to export extension")
    virtual void Release() noexcept {
        delete this;
    }

protected:
    virtual ~IExtension() = default;
};

/**
 * @brief A shared pointer to a IExtension interface
 */
using IExtensionPtr = std::shared_ptr<IExtension>;

/**
 * @brief Creates the default instance of the extension
 *
 * @param ext Extension interface
 */
INFERENCE_EXTENSION_API(void) CreateExtensionShared(IExtensionPtr& ext);

/**
 * @note: Deprecated API
 * @brief Creates the default instance of the extension
 * @param ext Extension interface
 * @param resp Responce
 * @return InferenceEngine::OK if extension is constructed and InferenceEngine::GENERAL_ERROR otherwise
 */
#if defined(_WIN32)
INFERENCE_ENGINE_DEPRECATED("Use IE_DEFINE_EXTENSION_CREATE_FUNCTION macro")
INFERENCE_EXTENSION_API(StatusCode)
CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept;
#else
INFERENCE_EXTENSION_API(StatusCode)
CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept INFERENCE_ENGINE_DEPRECATED("Use IE_DEFINE_EXTENSION_CREATE_FUNCTION macro");
#endif

/**
 * @def IE_DEFINE_EXTENSION_CREATE_FUNCTION
 * @brief Generates extension creation function
 */
#define IE_DEFINE_EXTENSION_CREATE_FUNCTION(ExtensionType)                                                                  \
INFERENCE_EXTENSION_API(void) InferenceEngine::CreateExtensionShared(std::shared_ptr<InferenceEngine::IExtension>& ext) {   \
    ext = std::make_shared<ExtensionType>();                                                                                    \
}
}  // namespace InferenceEngine
