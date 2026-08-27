// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ipf_client.hpp"

#ifdef OV_AUTO_ENABLE_IPF

#    include <vector>

#    include "ClientApiC.h"
#    include "log_util.hpp"

namespace ov {
namespace auto_plugin {
namespace device_monitor {

inline std::string get_log_tag() {
    return "[IPF]";
}

namespace {

using IpfQueryFn = ipf_err_t (*)(void*, const char*, char*, size_t*);

// Query IPF node/value data with the two-call buffer-size protocol.
std::string query_ipf_string(void* handle, IpfQueryFn query_fn, const std::string& path) {
    size_t len = 0;
    ipf_err_t status = query_fn(handle, path.c_str(), nullptr, &len);
    if (status != IpfError::IPF_ERR_BUFFERTOOSMALL || len == 0) {
        LOG_WARNING_TAG("IpfClientApiAdapter: IPF query(%s) size query failed: %s", path.c_str(), ipf_ef_error_str(status));
        return {};
    }
    std::vector<char> buf(len);
    status = query_fn(handle, path.c_str(), buf.data(), &len);
    if (status != IpfError::IPF_ERR_OK) {
        LOG_WARNING_TAG("IpfClientApiAdapter: IPF query(%s) failed: %s", path.c_str(), ipf_ef_error_str(status));
        return {};
    }
    std::string result(buf.data(), len);
    if (!result.empty() && result.back() == '\0') {
        result.pop_back();
    }
    return result;
}

void ipf_event_trampoline(const char* path, const char* event, void* context) {
    if (event == nullptr) {
        LOG_WARNING_TAG("IpfClientApiAdapter: received null event payload from %s", path != nullptr ? path : "<null>");
        return;
    }
    LOG_DEBUG_TAG("IpfClientApiAdapter: received event from %s. Event data: %s",
                  path != nullptr ? path : "<null>",
                  event);
    auto* callback = static_cast<IpfEventCallback*>(context);
    if (callback != nullptr && *callback) {
        (*callback)(event);
    }
}

}  // namespace

IpfClientApiAdapter::IpfClientApiAdapter() {
    const ipf_err_t status = IpfCreate(nullptr, &m_handle);
    if (status != IpfError::IPF_ERR_OK) {
        m_handle = nullptr;
        LOG_WARNING_TAG("IpfClientApiAdapter: IPF ClientApi initialization failed: %s (%s)",
                        ipf_ef_error_str(status),
                        IpfGetLastErrorMessage());
        return;
    }
    LOG_INFO_TAG("IpfClientApiAdapter: IPF ClientApi initialized successfully");
}

IpfClientApiAdapter::~IpfClientApiAdapter() {
    if (m_callback_context) {
        unregister_event(m_registered_path);
    }
    if (m_handle != nullptr) {
        IpfDestroy(m_handle);
        m_handle = nullptr;
    }
}

bool IpfClientApiAdapter::is_valid() const {
    return m_handle != nullptr;
}

std::string IpfClientApiAdapter::get_node(const std::string& path) {
    if (m_handle == nullptr) {
        return {};
    }
    return query_ipf_string(m_handle, &IpfGetNode, path);
}

std::string IpfClientApiAdapter::get_value(const std::string& path) {
    if (m_handle == nullptr) {
        return {};
    }
    return query_ipf_string(m_handle, &IpfGetValue, path);
}

bool IpfClientApiAdapter::register_event(const std::string& path, IpfEventCallback callback) {
    if (m_handle == nullptr) {
        return false;
    }
    if (m_callback_context) {
        // Reject re-registration while already active.
        LOG_WARNING_TAG("IpfClientApiAdapter: register_event(%s) rejected, already registered for %s",
                        path.c_str(),
                        m_registered_path.c_str());
        return false;
    }
    m_callback_context = std::make_unique<IpfEventCallback>(std::move(callback));
    const ipf_err_t status = IpfRegisterEvent(m_handle, path.c_str(), &ipf_event_trampoline, m_callback_context.get());
    if (status != IpfError::IPF_ERR_OK) {
        LOG_WARNING_TAG("IpfClientApiAdapter: failed to register for %s: %s", path.c_str(), ipf_ef_error_str(status));
        m_callback_context.reset();
        return false;
    }
    m_registered_path = path;
    LOG_INFO_TAG("IpfClientApiAdapter: registered for %s", path.c_str());
    return true;
}

void IpfClientApiAdapter::unregister_event(const std::string& path) {
    if (m_handle == nullptr || !m_callback_context) {
        return;
    }
    if (path != m_registered_path) {
        // Ignore unregister for a path other than the one registered.
        LOG_WARNING_TAG("IpfClientApiAdapter: unregister_event(%s) ignored, currently registered for %s",
                        path.c_str(),
                        m_registered_path.c_str());
        return;
    }
    const ipf_err_t status = IpfUnregisterEvent(m_handle, path.c_str(), &ipf_event_trampoline);
    if (status == IpfError::IPF_ERR_OK) {
        m_callback_context.reset();
        m_registered_path.clear();
    } else {
        // Leak the context; IPF may still hold a pointer to it.
        LOG_WARNING_TAG("IpfClientApiAdapter: IpfUnregisterEvent failed: %s, leaking callback context",
                        ipf_ef_error_str(status));
        (void)m_callback_context.release();
    }
}

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov

#endif  // OV_AUTO_ENABLE_IPF
