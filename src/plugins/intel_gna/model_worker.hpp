// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cstdint>
#include <memory>

#include <gna2-model-api.h>
#include "gna_device_interface.hpp"

namespace GNAPluginNS {
class Gna2ModelWrapper;

class ModelWorker {
public:
    virtual ~ModelWorker() = default;

    virtual const Gna2Model* model() const = 0;
    virtual Gna2Model* model() = 0;

    virtual void enqueue_request() = 0;

    // TODO define new enum
    virtual GNARequestWaitStatus wait(int64_t timeout_miliseconds) = 0;

    virtual bool is_free() const = 0;

    virtual uint32_t representing_index() const = 0;
    virtual void set_representing_index(uint32_t index) = 0;

    virtual void set_result(const InferenceEngine::BlobMap& result) = 0;
    virtual void set_result(InferenceEngine::BlobMap&& result) = 0;

    virtual InferenceEngine::BlobMap& result() = 0;
};

}  // namespace GNAPluginNS
