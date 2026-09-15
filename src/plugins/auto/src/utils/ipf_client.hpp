// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef MULTIUNITTEST
#define auto_plugin mock_auto_plugin
#endif

#ifdef OV_AUTO_ENABLE_IPF

#    include <functional>
#    include <memory>
#    include <string>

namespace ov {
namespace auto_plugin {
namespace device_monitor {

// Callback invoked with the raw JSON event payload delivered by IPF for a registered path.
using IpfEventCallback = std::function<void(const std::string& event_json)>;

// Abstracts the IPF ClientApi C ABI (ClientApiC.h) so TelemetryClient::Impl depends only on this
// interface instead of the raw C API, making IPF interactions mockable in unit tests.
class IIpfClient {
public:
    virtual ~IIpfClient() = default;

    // True if the underlying IPF client handle was created successfully.
    virtual bool is_valid() const = 0;

    // Query a node/value by path. Returns an empty string on failure (failure is already logged
    // by the implementation).
    virtual std::string get_node(const std::string& path) = 0;
    virtual std::string get_value(const std::string& path) = 0;

    // Registers for change notifications on path; returns whether registration succeeded.
    virtual bool register_event(const std::string& path, IpfEventCallback callback) = 0;
    // Best-effort unregister; safe to call even if register_event was never called or failed.
    virtual void unregister_event(const std::string& path) = 0;
};

// Real implementation: creates/destroys an IPF ClientApi handle and relays calls to it.
class IpfClientApiAdapter : public IIpfClient {
public:
    IpfClientApiAdapter();
    ~IpfClientApiAdapter() override;

    IpfClientApiAdapter(const IpfClientApiAdapter&) = delete;
    IpfClientApiAdapter& operator=(const IpfClientApiAdapter&) = delete;

    bool is_valid() const override;
    std::string get_node(const std::string& path) override;
    std::string get_value(const std::string& path) override;
    bool register_event(const std::string& path, IpfEventCallback callback) override;
    void unregister_event(const std::string& path) override;

private:
    void* m_handle = nullptr;
    // Owns the heap-allocated callback passed to IPF as the event context; freed only after a
    // confirmed successful unregister to avoid a use-after-free if IPF still holds a stale reference.
    std::unique_ptr<IpfEventCallback> m_callback_context;
    // Path currently registered for m_callback_context.
    std::string m_registered_path;
};

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov

#endif  // OV_AUTO_ENABLE_IPF
