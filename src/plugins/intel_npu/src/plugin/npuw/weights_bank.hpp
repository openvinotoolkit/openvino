// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "lazy_tensor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace weights {

class Bank {
public:
    explicit Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device)
        : m_core(core),
          m_alloc_device(alloc_device) {}

    // Allocate a new tensor (if needed) on a specified device. LazyTensor needs to be registered and evaluated first
    ov::Tensor get(const LazyTensor& tensor, const std::string& device);
    // Register LazyTensor in a bank if it's not there
    void registerLT(const LazyTensor& tensor, const std::string& device);
    // Evaluate and allocate all LazyTensors in the bank
    void evaluate_and_allocate();
    bool is_remote(const LazyTensor& tensor) const;

private:
    ov::Tensor unsafe_eval_and_alloc(const LazyTensor& tensor, const std::string& device);
    // Bank for specified device and their allocated memory
    std::unordered_map<std::string, std::unordered_map<LazyTensor, ov::Tensor, LazyTensor::Hash>> m_device_bank;
    std::mutex m_mutex;
    std::mutex m_alloc_mutex;
    std::shared_ptr<const ov::ICore> m_core = nullptr;
    std::shared_ptr<ov::IRemoteContext> m_remote_ctx = nullptr;
    std::string m_alloc_device;
};

std::shared_ptr<Bank> bank(const std::string& bank_name,
                           const std::shared_ptr<const ov::ICore>& core,
                           const std::string& alloc_device);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
