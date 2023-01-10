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
    MultiRemoteContext(const std::string pluginName) : m_plugin_name(pluginName) {}

    RemoteBlob::Ptr CreateBlob(const TensorDesc& tensorDesc, const ParamMap& params = {}) override {
        std::lock_guard<std::mutex> locker(m_mutex);
        if (m_contexts.size() == 0) {
            IE_THROW() << "no valid context available";
        }
        return m_contexts[0]->CreateBlob(tensorDesc, params);
    }

    ParamMap getParams() const override {
        std::lock_guard<std::mutex> locker(m_mutex);
        if (m_contexts.size() == 0) {
            IE_THROW() << "no valid context available";
        }
        return m_contexts[0]->getParams();
    }

    std::shared_ptr<RemoteContext> GetTargetContext(const std::string deviceName) {
        RemoteContext::Ptr res;
        DeviceIDParser parser(deviceName);
        std::string deviceIDLocal = parser.getDeviceID();
        if (deviceIDLocal.empty())
            deviceIDLocal = m_default_device_id;
        std::lock_guard<std::mutex> locker(m_mutex);
        for (auto&& iter : m_contexts) {
            if (iter->getDeviceName() == parser.getDeviceName() + "." + deviceIDLocal) {
                res = iter;
                break;
            }
        }
        return res;
    }

    std::string getDeviceName() const noexcept override {
        std::lock_guard<std::mutex> locker(m_mutex);
        return m_plugin_name;
    }

    void updateDeviceName(const std::string& devicename) {
        std::lock_guard<std::mutex> locker(m_mutex);
        m_plugin_name = devicename;
    }

    void AddContext(RemoteContext::Ptr hwcontext) {
        std::lock_guard<std::mutex> locker(m_mutex);
        if (hwcontext) {
            m_contexts.push_back(hwcontext);
        }
    }

    ~MultiRemoteContext() {
        m_contexts.clear();
    }

private:
    std::vector<RemoteContext::Ptr> m_contexts;
    std::string m_default_device_id = "0";
    std::string m_plugin_name;
    mutable std::mutex m_mutex;
};
}  // namespace MultiDevicePlugin
