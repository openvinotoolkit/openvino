// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-inference-api.h>

#include <cstdint>

#include "request_status.hpp"

enum Gna2AccelerationMode;
struct Gna2Model;

namespace ov {
namespace intel_gna {

// Interface name is different to the file naem due the lagacy reason.
// 1. Implementation file names should be changed in next PR.
// 2. Implementation of interface should be moved to ov::intel_gna namespace

/**
 * @interface Interface for invoking operation on GNA device.
 */
class GNADevice {
public:
    /**
     * @brief Destruct {GNADevice} object
     */
    virtual ~GNADevice() = default;

    /**
     * @brief Create gna model on device.
     * @param gna_model gna modle to by created on device
     * @return model id on device
     * @throw Exception in case of error
     */
    virtual uint32_t createModel(Gna2Model& gnaModel) const = 0;

    /**
     * @brief create request configuration for give id of model
     * @param modelID id of model on device
     * @return id of configuration for model
     * @throw Exception in case of error
     */
    virtual uint32_t createRequestConfig(const uint32_t modelID) const = 0;

    /**
     * @brief Add request to the execution queue.
     * @param requestConfigID id of request configuration to be used for equing request
     * @param gna2AccelerationMode acceleration mode of GNA device
     * @return enqueued request id on device
     * @throw Exception in case of error
     */
    virtual uint32_t enqueueRequest(const uint32_t requestConfigID, Gna2AccelerationMode gna2AccelerationMode) = 0;

    /**
     * @brief Wait for request to be finished.
     * @param requestID id of request enqueued on device
     * @param timeoutMilliseconds maximum timeout to be used for waiting
     * @return status of request given to the methoid. @see RequestStatus.
     * @throw Exception in case of error
     */
    virtual RequestStatus waitForRequest(uint32_t requestID, int64_t timeoutMilliseconds) = 0;

    /**
     * @brief Return maximum number of layers supported by device.
     * @return maximum layers count
     **/
    virtual uint32_t maxLayersCount() const = 0;

    /**
     * @brief close the device.
     **/
    virtual void close() {}
};

}  // namespace intel_gna
}  // namespace ov
