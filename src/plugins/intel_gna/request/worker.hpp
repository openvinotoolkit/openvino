// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-model-api.h>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cstdint>
#include <memory>

#include "request_status.hpp"

namespace GNAPluginNS {
namespace request {

class ModelWrapper;

class Worker {
public:
    virtual ~Worker() = default;

    virtual const Gna2Model* model() const = 0;
    virtual Gna2Model* model() = 0;

    virtual void enqueue_request() = 0;

    virtual RequestStatus wait(int64_t timeout_miliseconds) = 0;

    virtual bool is_free() const = 0;

    virtual uint32_t representing_index() const = 0;
    virtual void set_representing_index(uint32_t index) = 0;

    virtual void set_result(const InferenceEngine::BlobMap& result) = 0;
    virtual void set_result(InferenceEngine::BlobMap&& result) = 0;

    virtual InferenceEngine::BlobMap& result() = 0;
};

}  // namespace request
}  // namespace GNAPluginNS
