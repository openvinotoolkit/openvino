// Copyright (C) 2024-2025 Intel Corporation
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
// Forward declaration
class LLMCompiledModel;
class CompiledModel;
namespace weights {

class Bank {
public:
    explicit Bank(const std::shared_ptr<const ov::ICore>& core,
                  const std::string& alloc_device,
                  const std::string& bank_name)
        : m_core(core),
          m_alloc_device(alloc_device),
          m_bank_name(bank_name) {}

    // Register LazyTensor in a bank if it's not there. Returns LazyTensor's unique id
    int64_t registerLT(const LazyTensor& tensor, const std::string& device);

    // Get registered, allocated and evaluated tensor on a specified device
    ov::Tensor get(int64_t uid, const std::string& device);

    // Evaluate and allocate all LazyTensors in the bank
    void evaluate_and_allocate();

    bool is_remote(int64_t uid) const;

    std::string get_name() const;

private:
    friend class ov::npuw::LLMCompiledModel;
    friend class ov::npuw::CompiledModel;

    struct StoredTensor {
        LazyTensor lt;
        ov::Tensor tensor;
    };
    // Bank for specified device and their allocated memory
    struct DeviceBank {
        std::unordered_map<int64_t, StoredTensor> storage;
        std::unordered_map<LazyTensor, int64_t, LazyTensor::Hash> registered_tensors;
        mutable std::mutex mutex;
    };
    std::unordered_map<std::string, DeviceBank> m_device_banks;

    void serialize(std::ostream& stream) const;
    static std::shared_ptr<Bank> deserialize(std::istream& stream,
                                             const std::shared_ptr<const ov::ICore>& core,
                                             const std::string& name);
    // Used during deserialization
    void read_and_add_tensor(std::istream& stream, int64_t uid, const std::string& device);

    mutable std::mutex m_mutex;
    std::shared_ptr<const ov::ICore> m_core = nullptr;
    std::string m_alloc_device;
    int64_t uid_count = 0;
    std::string m_bank_name;
};

std::shared_ptr<Bank> bank(const std::string& bank_name,
                           const std::shared_ptr<const ov::ICore>& core,
                           const std::string& alloc_device);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
