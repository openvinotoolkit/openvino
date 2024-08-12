// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
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
    Bank(const std::shared_ptr<const ov::IPlugin>& plugin) : m_plugin(plugin){};

    // Capture CPU version of the tensor
    ov::Tensor update(const ov::Tensor& tensor);

    // Based on previously captured tensor allocate a new tensor (if needed) on a specified device
    ov::Tensor get(const ov::Tensor& tensor, const std::string& device);

private:
    // Default CPU bank. Filled by update()
    std::unordered_map<void*, ov::Tensor> m_bank;
    std::unordered_map<std::string, std::unordered_map<void*, ov::Tensor>> m_device_bank;
    std::mutex m_mutex;
    std::shared_ptr<const ov::IPlugin> m_plugin = nullptr;
    std::shared_ptr<ov::IRemoteContext> m_remote_ctx = nullptr;
};

std::shared_ptr<Bank> bank(const std::string& bank_name, const std::shared_ptr<const ov::IPlugin>& plugin);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
