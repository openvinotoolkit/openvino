// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include "logging.hpp"
#include "util.hpp"

using ov::npuw::weights::Bank;
using ov::npuw::weights::LazyTensor;

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
    std::shared_ptr<Bank> getBank(const std::string& bank_name,
                                  const std::shared_ptr<const ov::ICore>& core,
                                  const std::string& alloc_device);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

ov::Tensor Bank::get(const LazyTensor& tensor, const std::string& device) {
    if (device != "CPU" && device != "NPU") {
        OPENVINO_THROW("Unsupported device in weights bank allocation: ", device);
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    // Check if already allocated and transformed
    auto& device_bank = m_device_bank[device];
    auto iter_device = device_bank.find(tensor);
    if (iter_device == device_bank.end()) {
        ov::Tensor transformed_tensor = tensor.eval();

        if (device == "CPU" || m_alloc_device != device) {
            // No allocation - store as is
            device_bank[tensor] = transformed_tensor;
            return transformed_tensor;
        }

        // Allocation needed
        m_remote_ctx = m_core->get_default_context(device)._ptr;
        auto remote_tensor =
            m_remote_ctx->create_host_tensor(transformed_tensor.get_element_type(), transformed_tensor.get_shape());
        auto allocated_tensor = ov::make_tensor(remote_tensor);
        transformed_tensor.copy_to(allocated_tensor);
        device_bank[tensor] = allocated_tensor;
        return allocated_tensor;
    }

    return iter_device->second;
}

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name,
                                           const std::shared_ptr<const ov::ICore>& core,
                                           const std::string& alloc_device) {
    std::lock_guard<std::mutex> guard(m_mutex);

    auto iter = m_bank_map.find(bank_name);
    if (iter == m_bank_map.end()) {
        auto bank = std::make_shared<Bank>(core, alloc_device);
        m_bank_map[bank_name] = bank;
        return bank;
    }
    return iter->second.lock();
}

std::shared_ptr<Bank> ov::npuw::weights::bank(const std::string& bank_name,
                                              const std::shared_ptr<const ov::ICore>& core,
                                              const std::string& alloc_device) {
    if (bank_name.empty()) {
        // Don't share this bank in manager
        return std::make_shared<Bank>(core, alloc_device);
    }

    auto& instance = BankManager::getInstance();
    return instance.getBank(bank_name, core, alloc_device);
}
