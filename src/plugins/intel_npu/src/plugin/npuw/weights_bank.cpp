// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include "util.hpp"

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

ov::Tensor Bank::update(const std::shared_ptr<ov::op::v0::Constant>& node) {
    std::lock_guard<std::mutex> guard(m_mutex);

    auto tensor = ov::npuw::util::tensor_from_const(node);

    if (m_bank.find(tensor.data()) == m_bank.end()) {
        m_bank[tensor.data()] = node;
    }

    return tensor;
}

ov::Tensor Bank::get(const LazyTensor& tensor, const std::string& device) {
    if (device != "CPU" && device != "NPU") {
        OPENVINO_THROW("Unsupported device in weights bank allocation: ", device);
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    // Sanity check
    auto iter_cpu = m_bank.find(tensor.get_orig_data());
    if (iter_cpu == m_bank.end()) {
        OPENVINO_THROW("Unknown tensor in weights bank allocation!");
    }

    // Check if already allocated and transformed
    auto& device_bank = m_device_bank[device];
    auto iter_device = device_bank.find(tensor);
    if (iter_device != device_bank.end()) {
        return iter_device->second;
    }

    // Run transformations
    auto transformed_tensor = tensor.transform();
    device_bank[tensor] = transformed_tensor;

    if (device == "CPU") {
        return transformed_tensor;
    }

    // Allocation needed
    m_remote_ctx = m_core->get_default_context(device)._ptr;
    auto remote_tensor = m_remote_ctx->create_host_tensor(transformed_tensor.get_element_type(), transformed_tensor.get_shape());
    auto allocated_tensor = ov::make_tensor(remote_tensor);
    transformed_tensor.copy_to(allocated_tensor);
    device_bank[tensor] = allocated_tensor;

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
