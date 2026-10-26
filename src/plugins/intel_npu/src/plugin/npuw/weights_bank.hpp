// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "lazy_tensor.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "orc.hpp"
#include "orc/schema_npuw.hpp"

namespace ov {
namespace npuw {
namespace weights {
class Bank;
}  // namespace weights
namespace orc {
void serialize(ov::npuw::orc::Stream& stream, ov::npuw::weights::Bank& var);
}  // namespace orc
}  // namespace npuw
}  // namespace ov

namespace ov {
namespace npuw {
// Forward declaration
class LLMCompiledModel;
class CompiledModel;
namespace weights {

class WeightSharingContextAccessor {
public:
    WeightSharingContextAccessor(const std::shared_ptr<ov::weight_sharing::Context>& context) : m_context(context) {}

    struct Handle {
        friend WeightSharingContextAccessor;
        ~Handle() = default;
        Handle(Handle&&) noexcept = default;
        Handle& operator=(Handle&&) noexcept = default;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        
        std::shared_ptr<ov::weight_sharing::Context> context;
    private:
        Handle(std::mutex& m, std::shared_ptr<ov::weight_sharing::Context> ctx) : lock(m), context(ctx) {}
        std::lock_guard<std::mutex> lock;
    };

    Handle get_handle() {
        return Handle(m_mutex, m_context);
    }

    ov::weight_sharing::WeightBuffer request_shared_weight_source_buffer(size_t source_id, const element::Type& type, const ov::Shape& shape, const std::string &device, Handle &handle);
    void register_shared_weight_source_for_weight(ov::weight_sharing::WeightBuffer source_buffer,
                                                  size_t source_id,
                                                  const std::string& device,
                                                  Handle& handle);

    static ov::weight_sharing::WeightBuffer create_shared_buffer(int64_t source_id, std::size_t size);

    bool serialize_native_format(ov::npuw::orc::Stream& stream);
    bool serialize_native_format(ov::npuw::orc::Stream& stream, Handle& handle);
    bool deserialize_native_format(ov::npuw::orc::Stream& stream, const ov::ICore &core);
    bool deserialize_native_format(ov::npuw::orc::Stream& stream, const ov::ICore &core, Handle& handle);
protected:
    /* CVS-198039 WeightMetaData has no m_shape */
    const ov::op::v0::Constant& get_transformed_constant(size_t source_id, Handle& handle) const;

private:
    std::mutex m_mutex;
    std::shared_ptr<ov::weight_sharing::Context> m_context = nullptr;
    std::unordered_map<size_t, ov::weight_sharing::WeightBuffer> m_shared_sources;

    std::unordered_map<size_t, ov::op::v0::Constant> m_transformed_weights;
};


class Bank : public WeightSharingContextAccessor {
public:
    static constexpr ov::npuw::orc::TypeId kOrcType =
        static_cast<ov::npuw::orc::TypeId>(ov::npuw::orc::schema_npuw::WeightsBank::ID);
    // Version 0 is the frozen baseline on the wire. Any further layout changes
    // must be introduced through a new versioned payload rather than by mutating v0.
    static constexpr ov::npuw::orc::Version kOrcVersion = 0u;

    Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device, const std::string& bank_name,
         std::shared_ptr<ov::weight_sharing::Context> weight_sharing_context = nullptr);

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
    friend void ov::npuw::orc::serialize(ov::npuw::orc::Stream& stream, ov::npuw::weights::Bank& var);

    struct StoredTensor {
        LazyTensor lt;
        ov::Tensor tensor;
        bool is_evaluated() const;
    };
    // Bank for specified device and their allocated memory
    struct DeviceBank {
        std::unordered_map<int64_t, StoredTensor> storage;
        std::unordered_map<LazyTensor, int64_t, LazyTensor::Hash> registered_tensors;
    };
    std::unordered_map<std::string, DeviceBank> m_device_banks;

    void evaluate_cpu(DeviceBank& device_bank, const std::vector<LazyTensor>& to_process);
    void evaluate_and_allocate_on_device(DeviceBank& device_bank,
                                         const std::vector<LazyTensor>& to_process,
                                         const std::string& device);

    void serialize(ov::npuw::orc::Stream& stream);
    void deserialize(ov::npuw::orc::Stream& stream);
    void read_and_add_tensor(ov::npuw::orc::Stream& stream, int64_t uid, const std::string& device);

    mutable std::mutex m_mutex;
    std::shared_ptr<const ov::ICore> m_core = nullptr;
    std::string m_alloc_device;
    int64_t uid_count = 1;
    std::string m_bank_name;
};

std::shared_ptr<Bank> bank(const std::string& bank_name,
                           const std::shared_ptr<const ov::ICore>& core,
                           const std::string& alloc_device,
                           std::shared_ptr<ov::weight_sharing::Context> weight_shared_context = nullptr);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
