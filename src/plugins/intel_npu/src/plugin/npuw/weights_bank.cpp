// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include "logging.hpp"
#include "openvino/core/parallel.hpp"
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
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    if (device_for_alloc != "CPU" && device_for_alloc != "NPU") {
        OPENVINO_THROW("Unsupported device in weights bank allocation: ", device_for_alloc);
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    auto& device_bank = m_device_bank[device_for_alloc];
    auto iter_device = device_bank.find(tensor);

    // Should be already allocated at this point
    NPUW_ASSERT(iter_device != device_bank.end());

    return iter_device->second;
}

void Bank::registerLT(const LazyTensor& tensor, const std::string& device) {
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    if (device_for_alloc != "CPU" && device_for_alloc != "NPU") {
        OPENVINO_THROW("Unsupported device in weights bank allocation: ", device_for_alloc);
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    auto& device_bank = m_device_bank[device_for_alloc];
    if (device_bank.find(tensor) == device_bank.end()) {
        device_bank[tensor] = ov::Tensor();
    }
}

void Bank::evaluate_and_allocate() {
    std::lock_guard<std::mutex> guard(m_mutex);

    for (auto&& bank : m_device_bank) {
        const auto& device_for_alloc = bank.first;
        auto& device_bank = bank.second;
        std::vector<LazyTensor> vec;
        for (const auto& el : device_bank) {
            vec.push_back(el.first);
        }
        ov::parallel_for(vec.size(), [&](std::size_t idx) {
            const auto& lt = vec[idx];
            auto& transformed_tensor = device_bank[lt];
            if (!transformed_tensor) {
                transformed_tensor = lt.eval();
            }

            // Device name is known for each LT - set by registerLT
            if (device_for_alloc != "CPU") {
                // Allocate
                m_remote_ctx = m_core->get_default_context(device_for_alloc)._ptr;
                auto remote_tensor = m_remote_ctx->create_host_tensor(transformed_tensor.get_element_type(),
                                                                      transformed_tensor.get_shape());
                auto allocated_tensor = ov::make_tensor(remote_tensor);
                transformed_tensor.copy_to(allocated_tensor);
                transformed_tensor = allocated_tensor;
            }
        });
    }
}

bool Bank::is_remote(const LazyTensor& tensor) const {
    // FIXME: make generic
    if (m_device_bank.find("NPU") == m_device_bank.end()) {
        return false;
    }
    const auto& cpu_bank = m_device_bank.at("NPU");
    return cpu_bank.find(tensor) != cpu_bank.end();
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
