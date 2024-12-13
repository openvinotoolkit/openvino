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
// Forward declaration
class LLMCompiledModel;
class CompiledModel;
namespace weights {

class Bank {
public:
    explicit Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device)
        : m_core(core),
          m_alloc_device(alloc_device) {}

    // Allocate a new tensor (if needed) on a specified device. LazyTensor needs to be registered and evaluated first
    ov::Tensor get(const LazyTensor& tensor, const std::string& device);
    // Register LazyTensor in a bank if it's not there. Returns LazyTensor's unique id
    std::size_t registerLT(const LazyTensor& tensor, const std::string& device);
    // Evaluate and allocate all LazyTensors in the bank
    void evaluate_and_allocate();
    bool is_remote(const LazyTensor& tensor) const;

private:
    friend class ov::npuw::LLMCompiledModel;
    friend class ov::npuw::CompiledModel;

    // Bank for specified device and their allocated memory
    struct DeviceBank {
        // As value additionally uid for serialization
        std::unordered_map<LazyTensor, std::pair<ov::Tensor, std::size_t>, LazyTensor::Hash> storage;
        // This simplified storage is only used if the bank has been deserialized. bool specifies if the tensor has
        // already been allocated on the device
        std::unordered_map<std::size_t, std::pair<ov::Tensor, bool>> deserialized_storage;
        mutable std::mutex mutex;
    };
    std::unordered_map<std::string, DeviceBank> m_device_banks;

    ov::Tensor eval_and_alloc(const LazyTensor& tensor, DeviceBank& dbank, const std::string& device);

    void serialize(std::ostream& stream) const;
    static std::shared_ptr<Bank> deserialize(std::istream& stream, const std::shared_ptr<const ov::ICore>& core);
    // Used during deserialization
    void add_element(std::size_t uid, const ov::Tensor& tensor, const std::string& device);
    // Used with deserialized bank only. Always allocate on the device and copy data
    ov::Tensor get(std::size_t uid, const std::string& device);

    mutable std::mutex m_mutex;
    std::shared_ptr<const ov::ICore> m_core = nullptr;
    std::string m_alloc_device;
    std::size_t uid_count = 0;
};

std::shared_ptr<Bank> bank(const std::string& bank_name,
                           const std::shared_ptr<const ov::ICore>& core,
                           const std::string& alloc_device);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
