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
    std::shared_ptr<Bank> getBank(const std::string& bank_name);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

ov::Tensor Bank::get(const ov::Tensor& tensor) {
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

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name) {
    std::lock_guard<std::mutex> guard(m_mutex);

    // Extend ptr lifetime until obtained
    std::shared_ptr<Bank> bank = nullptr;
    if (m_bank_map.find(bank_name) == m_bank_map.end()) {
        bank = std::make_shared<Bank>();
        m_bank_map[bank_name] = bank;
    }

    return m_bank_map[bank_name].lock();
}

std::shared_ptr<Bank> ov::npuw::weights::bank(const std::string& bank_name) {
    if (bank_name.empty()) {
        // Don't share this bank in manager
        return std::make_shared<Bank>();
    }

    auto& instance = BankManager::getInstance();
    return instance.getBank(bank_name);
}
