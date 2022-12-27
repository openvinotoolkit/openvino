// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * OpenCL context and OpenCL shared memory blobs
 *
 * @file gpu_context_api_ocl.hpp
 */
#pragma once

#include <ie_remote_context.hpp>
#include <memory>
#include <string>

#include "ie_compound_blob.h"
#include "ie_core.hpp"

namespace MultiDevicePlugin {

/**
 * @brief This class represents an abstraction for Multi plugin remote context
 * which is wrapper of underline hardware remote contexts.
 * The plugin object derived from this class can be obtained either with
 * GetContext() method of Executable network or using CreateContext() Core call.
 */
class MultiRemoteContext : public RemoteContext {
public:
    /**
     * @brief A smart pointer to the MultiRemoteContext object
     */
    using Ptr = std::shared_ptr<MultiRemoteContext>;

    MultiRemoteContext() = default;

    MultiRemoteContext(const std::vector<RemoteContext::Ptr> contexts) {
        m_contexts = contexts;
    }

    RemoteBlob::Ptr CreateBlob(const TensorDesc& tensorDesc, const ParamMap& params = {}, int index = 0) {
        return m_contexts[index]->CreateBlob(tensorDesc, params);
    }

    RemoteBlob::Ptr CreateBlob(const TensorDesc& tensorDesc, const ParamMap& params = {}) override {
        return m_contexts[0]->CreateBlob(tensorDesc, params);
    }

    ParamMap getParams() const override {
        return m_contexts[0]->getParams();
    }

    std::pair<std::shared_ptr<RemoteContext>, std::shared_ptr<void>> GetTargetContext(const std::string deviceName) {
        RemoteContext::Ptr res;
        DeviceIDParser parser(deviceName);
        std::string deviceIDLocal = parser.getDeviceID();
        if (deviceIDLocal.empty())
            deviceIDLocal = m_default_device_id;
        for (auto&& iter : m_contexts) {
            if (iter->getDeviceName() == parser.getDeviceName() + "." + deviceIDLocal) {
                res = iter;
                break;
            }
        }
        return {res, m_libs[res->getDeviceName()]};
    }

    bool isEmpty() const {
        return m_contexts.size() == 0;
    }

    std::string getDeviceName() const noexcept override {
        std::string deviceName = "MULTI";
        return deviceName;
    };

    void AddContext(RemoteContext::Ptr hwcontext, std::shared_ptr<void> deplib) {
        if (hwcontext) {
            m_contexts.push_back(hwcontext);
            m_libs[hwcontext->getDeviceName()] = deplib;
        }
    }

    ~MultiRemoteContext() {
        m_contexts.clear();
    }

private:
    std::vector<RemoteContext::Ptr> m_contexts {nullptr};
    std::map<std::string, std::shared_ptr<void>> m_libs;
    std::string m_default_device_id = "0";
};
}  // namespace MultiDevicePlugin
