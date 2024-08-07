// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>
#include <unordered_map>

#include "openvino/openvino.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin.hpp"

namespace ov {
namespace npuw {
namespace weights_bank {

class WeightBank {
public:
    WeightBank(const std::string& device_name, const std::shared_ptr<const ov::IPlugin>& plugin) : m_device_name(device_name), m_plugin(plugin) {
        if (m_device_name != "CPU" && m_device_name != "NPU") {
            OPENVINO_THROW("NPUW doesn't support ", m_device_name, " device for weights sharing!");
        }
        if (m_device_name == "NPU" && !m_remote_ctx) {
            m_remote_ctx = m_plugin->get_core()->get_default_context("NPU")._ptr;
        }
    }

    ov::SoPtr<ov::Tensor> getSharedTensor(const ov::element::Type& type, const ov::Shape& shape, void* host_data_ptr) {
        if(!host_data_ptr) {
            OPENVINO_THROW("Fatal: nullptr in weights bank allocation!");
        }

        std::lock_guard<std::mutex> guard(m_mutex);

        // no special allocation needed
        if (m_weights_bank.count(host_data_ptr) > 0) { 
            return m_weights_bank.at(host_data_ptr);
        }

        // need to allocate first
        if (m_device_name == "CPU") {
            m_weights_bank[host_data_ptr] = std::make_shared<ov::Tensor>(type, shape, host_data_ptr);
        } else { // m_device_name == "NPU"
            auto remote_tensor = m_remote_ctx->create_host_tensor(type, shape);
            m_weights_bank[host_data_ptr] = std::make_shared<ov::Tensor>(ov::make_tensor(remote_tensor));
        }

        return m_weights_bank.at(host_data_ptr);
    }

private:
    // Note: suits both - remote and ordinary tensors
    std::unordered_map<void*, ov::SoPtr<ov::Tensor>> m_weights_bank;
    std::string m_device_name;
    std::shared_ptr<ov::IRemoteContext> m_remote_ctx = nullptr;
    const std::shared_ptr<const ov::IPlugin>& m_plugin = nullptr;
    std::mutex m_mutex;
};


class WeightsBankManager
{
public:
    static WeightsBankManager& getInstance()
    {
        static WeightsBankManager instance;
        return instance;
    }

private:
    WeightsBankManager() {}
    WeightsBankManager(WeightsBankManager const&);
    void operator=(WeightsBankManager const&);

public:
    // Public API
    std::shared_ptr<WeightBank> getBank(const std::string& bank_name, const std::string& device_name) {
        std::string bank_key = bank_name + device_name;
        if (m_weights_bank_map.count(bank_key) == 0) {
            m_weights_bank_map[bank_key] = std::make_shared<WeightBank>(device_name, m_plugin);
        }
        return m_weights_bank_map.at(bank_key);
    }

    void initPlugin(const std::shared_ptr<const ov::IPlugin>& plugin) {
        if (!m_plugin) {
            m_plugin = plugin;
        }
    }

private:
    // Data
    std::unordered_map<std::string, std::shared_ptr<WeightBank>> m_weights_bank_map;
    std::shared_ptr<const ov::IPlugin> m_plugin = nullptr;
};

} // namespace weights_bank
} // namespace npuw
} // namespace ov
