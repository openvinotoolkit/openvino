// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

using ov::npuw::weights::Bank;

class BankManager {
public:
    static BankManager& getInstance() {
        static BankManager instance;
        return instance;
    }

private:
    BankManager() {}
    BankManager(BankManager const&) = delete;
    void operator=(BankManager const&) = delete;

public:
    // Public API
    std::shared_ptr<Bank> getBank(const std::string& bank_name, const std::shared_ptr<const ov::ICore>& core);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

ov::Tensor Bank::update(const ov::Tensor& tensor) {
    if (!tensor) {
        OPENVINO_THROW("Uninitialized tensor in weights bank allocation!");
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    if (m_bank.find(tensor.data()) == m_bank.end()) {
        // need to allocate first
        m_bank[tensor.data()] = tensor;
    }

    return tensor;
}

ov::Tensor Bank::get(const ov::Tensor& tensor, const std::string& device) {
    if (!tensor) {
        OPENVINO_THROW("Uninitialized tensor in weights bank allocation!");
    }

    if (device != "CPU" && device != "NPU") {
        OPENVINO_THROW("Unsupported device in weights bank allocation: ", device);
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    auto iter_cpu = m_bank.find(tensor.data());
    if (iter_cpu == m_bank.end()) {
        OPENVINO_THROW("Unknown tensor in weights bank allocation!");
    }

    // If target device is CPU - just reuse the default bank
    if (device == "CPU") {
        return iter_cpu->second;
    }

    // Non-CPU - check if the tensor is already there
    auto& device_bank = m_device_bank[device];
    auto iter_device = device_bank.find(tensor.data());
    if (iter_device != device_bank.end()) {
        // Already allocated on the device - reuse
        return iter_device->second;
    }

    // Allocation needed
    m_remote_ctx = m_core->get_default_context(device)._ptr;
    auto remote_tensor = m_remote_ctx->create_host_tensor(tensor.get_element_type(), tensor.get_shape());
    auto allocated_tensor = ov::make_tensor(remote_tensor);
    tensor.copy_to(allocated_tensor);
    device_bank[tensor.data()] = allocated_tensor;
    return allocated_tensor;
}

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name, const std::shared_ptr<const ov::ICore>& core) {
    std::lock_guard<std::mutex> guard(m_mutex);

    auto iter = m_bank_map.find(bank_name);
    if (iter == m_bank_map.end()) {
        auto bank = std::make_shared<Bank>(core);
        m_bank_map[bank_name] = bank;
        return bank;
    }
    return iter->second.lock();
}

std::shared_ptr<Bank> ov::npuw::weights::bank(const std::string& bank_name,
                                              const std::shared_ptr<const ov::ICore>& core) {
    if (bank_name.empty()) {
        // Don't share this bank in manager
        return std::make_shared<Bank>(core);
    }

    auto& instance = BankManager::getInstance();
    return instance.getBank(bank_name, core);
}
