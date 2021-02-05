// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include <ie_blob.h>
#include <ie_common.h>
#include <ie_preprocess.hpp>

#include <map>
#include <memory>
#include <string>

namespace InferenceEngine {

/**
 * @interface IInferRequestInternal
 * @brief An internal API of synchronous inference request to be implemented by plugin,
 * which is used in InferRequestBase forwarding mechanism
 * @ingroup ie_dev_api_infer_request_api
 */
class IInferRequestInternal {
public:
    /**
     * @brief A shared pointer to a IInferRequestInternal interface
     */
    typedef std::shared_ptr<IInferRequestInternal> Ptr;

    /**
     * @brief Destroys the object.
     */
    virtual ~IInferRequestInternal() = default;

    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of IInferRequest while request is ongoing (running or waiting in queue)
     */
    virtual void Infer() = 0;

    /**
     * @brief Cancel current inference request execution
     */
    virtual StatusCode Cancel() = 0;

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     *  Note: not all plugins may provide meaningful data
     *  @return Returns a map of layer names to profiling information for that layer.
     */
    virtual std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const = 0;

    /**
     * @brief Set input/output data to infer
     * @note Memory allocation doesn't happen
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input
     * precision and size.
     */
    virtual void SetBlob(const std::string& name, const Blob::Ptr& data) = 0;

    /**
     * @brief Get input/output data to infer
     * @note Memory allocation doesn't happen
     * @param name - a name of input or output blob.
     * @return Returns input or output blob. The type of Blob must correspond to the network input
     * precision and size.
     */
    virtual Blob::Ptr GetBlob(const std::string& name) = 0;

    /**
     * @brief Sets pre-process for input data
     * @param name Name of input blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     * @param info Preprocess info for blob.
     */
    virtual void SetBlob(const std::string& name, const Blob::Ptr& data, const PreProcessInfo& info) = 0;

    /**
     * @brief Gets pre-process for input data
     * @param name Name of input blob.
     * @return Returns constant reference to PreProcessInfo structure
     */
    virtual const PreProcessInfo& GetPreProcess(const std::string& name) const = 0;

    /**
     * @brief Sets new batch size when dynamic batching is enabled in executable network that created this request.
     * @param batch - new batch size to be used by all the following inference calls for this request.
     */
    virtual void SetBatch(int batch) = 0;

    /**
     * @brief Queries memory states.
     * @return Returns memory states
     */
    virtual std::vector<IVariableStateInternal::Ptr> QueryState() = 0;
};

}  // namespace InferenceEngine
