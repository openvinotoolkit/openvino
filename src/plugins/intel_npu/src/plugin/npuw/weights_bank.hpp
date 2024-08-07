// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>
#include <tuple>
#include <unordered_map>

#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace weights {

class Bank {
public:
    Bank(const std::string& device_name);

    ov::Tensor get(const ov::Tensor& tensor);

private:
    // Note: suits both - remote and ordinary tensors
    std::unordered_map<void*, ov::Tensor> m_bank;
    std::string m_device_name;
    std::mutex m_mutex;
};

class BankManager {
public:
    static BankManager& getInstance() {
        static BankManager instance;
        return instance;
    }

private:
    BankManager() {}
    BankManager(BankManager const&);
    void operator=(BankManager const&);

public:
    // Public API
    std::shared_ptr<Bank> getBank(const std::string& bank_name, const std::string& device_name);

private:
    // Data
    std::unordered_map<std::string, std::shared_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

std::shared_ptr<Bank> bank(const std::string& bank_name, const std::string& device_name);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
