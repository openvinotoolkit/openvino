// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <gna2-inference-api.h>

#include "request_status.hpp"

enum Gna2AccelerationMode;
class Gna2Model;

// Interface name is different to the file naem due the lagacy reason.
// 1. Implementation file names should be changed in next PR.
// 2. Interface and Implementation should be moved to GNAPluginNS namespace

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
     * @param gan_model gna modle to by created on device
     * @return model id
     */
    virtual uint32_t create_model(Gna2Model& gna_model) const = 0;
    virtual uint32_t create_request_config(const uint32_t model_id) const = 0;
    virtual uint32_t max_layers_count() const = 0;
    virtual uint32_t enqueue_request(const uint32_t requestConfigId, Gna2AccelerationMode gna2AccelerationMode) = 0;
    virtual GNAPluginNS::RequestStatus wait_for_reuqest(uint32_t id, int64_t millisTimeout) = 0;
};
