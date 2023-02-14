// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-model-api.h>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cstdint>
#include <memory>

#include "request_status.hpp"

namespace ov {
namespace intel_gna {
namespace request {

class ModelWrapper;

/**
 * @interface Interface allowing to execute request for represented model in execution environment.
 */
class Worker {
public:
    /**
     * @brief Destroy {Worker} object
     */
    virtual ~Worker() = default;

    /**
     * @brief Return pointer to gna model represented by worker
     */
    virtual Gna2Model* model() = 0;

    /**
     * @brief Return pointer to gna model represented by worker
     */
    virtual const Gna2Model* model() const = 0;

    /**
     * @brief Enqueue request to requests queue for contained model.
     * @return true in case subrequest was properly enqueued, otherwise return false
     */
    virtual bool enqueueRequest() = 0;

    /**
     * @brief Wait untril request will be not finished for give timeout.
     * @param timeoutMilliseconds timeout in milliseconds
     * @return status of execution of ongoing request. @see RequestStatus
     */
    virtual RequestStatus wait(int64_t timeoutMilliseconds) = 0;

    /**
     * @brief Return true if worker is free and can used for enqueueing new request.
     */
    virtual bool isFree() const = 0;

    /**
     * @brief Return number of representing index. Can be used for identification.
     */
    virtual uint32_t representingIndex() const = 0;

    /**
     * @brief Set representing index.
     * @param index value to set for representing index
     */
    virtual void setRepresentingIndex(uint32_t index) = 0;

    /**
     * @brief return reference to result object for gna model represented by worker
     */
    virtual InferenceEngine::BlobMap& result() = 0;

    /**
     * @brief Set result object configuration for gna model represented by worker
     * @param result Refrence to object represetning result. @see InferenceEngine::BlobMap
     */
    virtual void setResult(const InferenceEngine::BlobMap& result) = 0;

    /**
     * @brief Set result object configuration for gna model represented by worker
     * @param result Refrence to object represetning result. @see InferenceEngine::BlobMap
     */
    virtual void setResult(InferenceEngine::BlobMap&& result) = 0;
};

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
