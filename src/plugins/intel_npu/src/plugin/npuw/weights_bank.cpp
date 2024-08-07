// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

using ov::npuw::weights::Bank;
using ov::npuw::weights::BankManager;

Bank::Bank(const std::string& device_name) : m_device_name(device_name) {
    if (m_device_name != "CPU") {
        OPENVINO_THROW("NPUW doesn't support ", m_device_name, " device for weights sharing!");
    }
}

ov::Tensor Bank::get(const ov::Tensor& tensor) {
    if (!tensor) {
        OPENVINO_THROW("Uninitialized tensor in weights bank allocation!");
    }

    void* host_data_ptr = tensor.data();

    std::lock_guard<std::mutex> guard(m_mutex);

    if (m_bank.find(host_data_ptr) == m_bank.end()) {
        // need to allocate first
        m_bank[host_data_ptr] = tensor;
    }

    return m_bank[host_data_ptr];
}

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name, const std::string& device_name) {
    std::lock_guard<std::mutex> guard(m_mutex);

    auto bank_key = bank_name + device_name;
    if (m_bank_map.find(bank_key) == m_bank_map.end()) {
        m_bank_map[bank_key] = std::make_shared<Bank>(device_name);
    }

    return m_bank_map[bank_key];
}

std::shared_ptr<Bank> ov::npuw::weights::bank(const std::string& bank_name, const std::string& device_name) {
    return BankManager::getInstance().getBank(bank_name, device_name);
}
