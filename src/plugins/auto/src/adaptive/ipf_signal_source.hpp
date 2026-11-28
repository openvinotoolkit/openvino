// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../utils/device_telemetry.hpp"
#include "observation.hpp"

namespace ov {
namespace auto_plugin {

// The IPF client shared with the synchronous queries of the plugin; the mutex keeps them from overlapping.
class IpfSignalSource : public ISignalSource {
public:
    IpfSignalSource(device_monitor::TelemetryClient& client, std::mutex& mutex) : m_client(client), m_mutex(mutex) {}
    Utilizations query(const DeviceQuery& devices) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_client.utilizations(devices);
    }

private:
    device_monitor::TelemetryClient& m_client;
    std::mutex& m_mutex;
};

}  // namespace auto_plugin
}  // namespace ov
