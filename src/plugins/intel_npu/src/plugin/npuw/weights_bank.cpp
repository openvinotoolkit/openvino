// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include "logging.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "serialization.hpp"
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
    BankManager(const BankManager&) = delete;
    void operator=(const BankManager&) = delete;

public:
    // Public API
    std::shared_ptr<Bank> getBank(const std::string& bank_name,
                                  const std::shared_ptr<const ov::ICore>& core,
                                  const std::string& alloc_device,
                                  std::shared_ptr<ov::weight_sharing::Context> weight_shared_context = nullptr);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};



ov::weight_sharing::WeightBuffer ov::npuw::weights::WeightSharingContextAccessor::create_shared_buffer(
    int64_t source_id,
    std::size_t size) {
    // Create page aligned buffer to satisfy further GPU-NPU memory sharing requirements
    auto raw = std::make_shared<ov::AlignedBuffer>(size, ov::util::get_system_page_size());
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            raw->get_ptr<char>(),
            raw->size(),
            raw,
            ov::create_base_descriptor(source_id, 0, raw));
}

ov::weight_sharing::WeightBuffer ov::npuw::weights::WeightSharingContextAccessor::request_shared_weight_source_buffer(
    size_t source_id,
    size_t size,
    const std::string& device,
    Handle& handle) {
    if (!handle.context) {
        return {};
    }

    auto shared_weight_source = ov::weight_sharing::get_runtime_weight_source(*handle.context, source_id);
    if (auto shared_buffer = shared_weight_source.m_weights.lock()) {
        OPENVINO_ASSERT(shared_weight_source.m_device == device,
                        "Shared weight source exists for the wrong device, expected: ",
                        device,
                        " actual: ",
                        shared_weight_source.m_device);
                m_shared_sources[source_id] = shared_buffer;
        return shared_buffer;
    }

    OPENVINO_ASSERT(shared_weight_source.m_device.empty() || shared_weight_source.m_device == device,
                    "Shared weight source exists for the wrong device, expected: ",
                    device,
                    " actual: ",
                    shared_weight_source.m_device);
    auto shared_buffer = WeightSharingContextAccessor::create_shared_buffer(source_id, size);
    register_shared_weight_source_for_weight(shared_buffer, source_id, device, handle);
    return shared_buffer;
}

void ov::npuw::weights::WeightSharingContextAccessor::register_shared_weight_source_for_weight(
    ov::weight_sharing::WeightBuffer source_buffer,
    size_t source_id,
    const std::string& device,
    Handle& handle) {
    m_shared_sources[source_id] = source_buffer;
    ov::weight_sharing::set_runtime_weight_source(*handle.context, source_buffer, device);
}

Bank::Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device, const std::string& bank_name, 
           std::shared_ptr<ov::weight_sharing::Context> weight_sharing_ctx)
    : WeightSharingContextAccessor(weight_sharing_ctx),
      m_core(core),
      m_alloc_device(alloc_device),
      m_bank_name(bank_name) {
    if (m_bank_name.empty()) {
        auto unique_name = ov::npuw::util::generate_random_string();
        LOG_WARN("Got an empty name for weights bank! Using a uniquely generated instead: " << unique_name);
        m_bank_name = unique_name;
    }
}

int64_t Bank::registerLT(const LazyTensor& tensor, const std::string& device) {
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks[device_for_alloc];

    auto iter_registered = device_bank.registered_tensors.find(tensor);
    if (iter_registered == device_bank.registered_tensors.end()) {
        auto uid = uid_count++;
        device_bank.registered_tensors[tensor] = uid;
        device_bank.storage[uid] = {tensor, ov::Tensor()};
        return uid;
    } else {
        // Already registered - can be safely detach the incoming tensor
        const_cast<LazyTensor&>(tensor).detach();
    }

    return iter_registered->second;
}

ov::Tensor Bank::get(int64_t uid, const std::string& device) {
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks.at(device_for_alloc);
    auto iter_device = device_bank.storage.find(uid);

    NPUW_ASSERT(iter_device != device_bank.storage.end() && iter_device->second.tensor &&
                "Tensor should be registered and allocated first!");

    return iter_device->second.tensor;
}

struct TensorToAllocate {
    LazyTensor::Meta meta;
    ov::Tensor allocated_tensor;
    int64_t uid;
};

void Bank::evaluate_and_allocate() {
    std::unique_lock guard(m_mutex);

    for (auto&& bank : m_device_banks) {
        const auto& device_for_alloc = bank.first;
        auto& device_bank = bank.second;

        std::vector<LazyTensor> to_process;
        to_process.reserve(device_bank.storage.size());
        for (const auto& el : device_bank.storage) {
            // Add non-allocated tensors for furter evaluation and allocation
            if (!el.second.tensor) {
                to_process.push_back(el.second.lt);
            }
        }

        if (device_for_alloc == "CPU") {
            evaluate_cpu(device_bank, to_process);
        } else {
            evaluate_and_allocate_on_device(device_bank, to_process, device_for_alloc);
        }
    }  // for (m_device_banks)
}

void Bank::evaluate_cpu(Bank::DeviceBank& device_bank, const std::vector<LazyTensor>& to_process) {
    // Note: not locking here. This is a private function, so Bank should handle the locks around it
    // as we lock in evaluate_and_allocate() now.
    ov::parallel_for(to_process.size(), [&](std::size_t idx) {
        const auto& lt = to_process[idx];
        auto iter_device_registered = device_bank.registered_tensors.find(lt);
        NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                    "Tensor should be registered first!");
        auto uid = iter_device_registered->second;
        auto t = lt.eval();

        auto& stored_tensor = device_bank.storage.at(uid);

        // Each LazyTensor describes either a shared `constant` or a shared `transformation` of such constants.
        // "Shared" means that, once allocated, the tensor's memory may be reused across different compiled-model instances
        // if the same `transformation` is repeated.
        // If other models repeat the same `transformations`, the LazyTensor representing a given `transformation`
        // will have the same UID in the `Bank`.
        // Encountering the same UID while compiling another model using the same `Bank` means that the `transformation`
        // has already been encountered and its memory may already be allocated.
        // The memory for a shared `transformation` is allocated as follows:
        // 1. Allocate memory for the shared `transformation`, aligned to the page size for subsequent use.
        // 2. Register the allocated memory as a `source_id` in `ov::weight_sharing::Context`.
        //    The `source_id` has the same value as the UID of the shared `transformation` represented by the LazyTensor.
        // 3. Create an `ov::Tensor` using the allocated memory.
        // `WeightSharingContextAccessor` only introduces step 2, while encapsulating step 1 internally.
        // `WeightSharingContextAccessor` owns and manages access to the shared weight memory.
        // It also provides access to `ov::weight_sharing::Context`.
        auto handle = this->get_handle();
        auto size = ov::util::get_memory_size_safe(t.get_element_type(), t.get_shape());
        OPENVINO_ASSERT(size.has_value(), "LazyTensor size overflow, uid: ", uid, " device: CPU");
        auto shared_source = this->request_shared_weight_source_buffer(uid, *size, "CPU", handle);

        if (shared_source) {
            stored_tensor.tensor = ov::Tensor(t.get_element_type(), t.get_shape(), shared_source->get_ptr());
        } else {
            stored_tensor.tensor = ov::Tensor(t.get_element_type(), t.get_shape());
        }

        // Get ownership of the weights, might be a mmaped object during import
        t.copy_to(stored_tensor.tensor);
        const_cast<LazyTensor&>(lt).detach();
    });
}

// Note: there are no locks in this function's parallel_for
// At this point all the LazyTensor->Tensor pairs in the map are already
// allocated and there are no conflicting reads/writes since all the tensors are unique.

// FIXME: this whole flow could be improved for the same bank
// processing from different threads. We could separate LazyTensors
// evaluation and the bank access. But it requires additional rework.
void Bank::evaluate_and_allocate_on_device(Bank::DeviceBank& device_bank,
                                           const std::vector<LazyTensor>& to_process,
                                           const std::string& device) {
    // Note: not locking here. This is a private function, so Bank should handle the locks around it
    // as we lock in evaluate_and_allocate() now.
    std::vector<TensorToAllocate> uids_to_allocated;
    uids_to_allocated.reserve(uid_count);

    for (const auto& lt : to_process) {
        auto iter_device_registered = device_bank.registered_tensors.find(lt);
        NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                    "Tensor should be registered first!");
        auto uid = iter_device_registered->second;
        uids_to_allocated.push_back({lt.eval_meta(), ov::Tensor(), uid});
    }
    // Sort by UIDs, lowest first
    std::sort(uids_to_allocated.begin(),
              uids_to_allocated.end(),
              [](const TensorToAllocate& a, const TensorToAllocate& b) {
                  return a.uid < b.uid;
              });

    // Allocate memory sequentially - in order of UID
    auto remote_ctx = m_core->get_default_context(device)._ptr;
    for (auto&& allocated : uids_to_allocated) {
        const auto evaluated_byte_size = ov::util::get_memory_size_safe(allocated.meta.type, allocated.meta.shape);
        OPENVINO_ASSERT(evaluated_byte_size.has_value(), "LazyTensor size overflow, uid: ", allocated.uid);

        // Each LazyTensor describes either a shared `constant` or a shared `transformation` of such constants.
        // "Shared" means that, once allocated, the tensor's memory may be reused across different compiled-model instances
        // if the same `transformation` is repeated.
        // If other models repeat the same `transformations`, the LazyTensor representing a given `transformation`
        // will have the same UID in the `Bank`.
        // Encountering the same UID while compiling another model using the same `Bank` means that the `transformation`
        // has already been encountered and its memory may already be allocated.
        // The memory for a shared `transformation` is allocated as follows:
        // 1. Allocate memory for the shared `transformation`, aligned to the page size for subsequent use.
        // 2. Register the allocated memory as a `source_id` in `ov::weight_sharing::Context`.
        //    The `source_id` has the same value as the UID of the shared `transformation` represented by the LazyTensor.
        // 3. Provided that these steps are successful, import RemoteTensor from the remote context using the allocated memory.
        // 4. Otherwise, assuming that ov::weight_sharing::Context is omitted, fallback to the original RemoteTensor creation mechanism.
        // `WeightSharingContextAccessor` only introduces step 2, while encapsulating step 1 internally.
        // `WeightSharingContextAccessor` owns and manages access to the shared weight memory.
        // It also provides access to `ov::weight_sharing::Context`.
        auto handle = this->get_handle();
        auto weight_shared_source_buffer = this->request_shared_weight_source_buffer(allocated.uid, *evaluated_byte_size, device, handle);

        ov::Tensor tensor;
        if (weight_shared_source_buffer) {
            ov::AnyMap params{
                ov::intel_npu::mem_type(ov::intel_npu::MemType::CPU_VA),
                ov::intel_npu::mem_handle(weight_shared_source_buffer->get_ptr()),
                ov::intel_npu::tensor_type(ov::intel_npu::TensorType::BINDED),
            };
            tensor = ov::make_tensor(remote_ctx->create_tensor(allocated.meta.type, allocated.meta.shape, params));
        } else {
            tensor = ov::make_tensor(remote_ctx->create_host_tensor(allocated.meta.type, allocated.meta.shape));
        }
        allocated = {allocated.meta, std::move(tensor), allocated.uid};
    }

    // Evaluate and copy into the device memory
    ov::parallel_for(uids_to_allocated.size(), [&](std::size_t idx) {
        auto& allocated = uids_to_allocated[idx];
        auto& stored_tensor = device_bank.storage.at(allocated.uid);

        auto transformed = stored_tensor.lt.eval();
        transformed.copy_to(allocated.allocated_tensor);
        stored_tensor.tensor = std::move(allocated.allocated_tensor);

        // Detach the evaluated LazyTensor from its memory here - when it is 100%
        // not needed anymore (transformations, if any, and copies are done)
        // Note: this is the non-CPU path!
        const_cast<LazyTensor&>(stored_tensor.lt).detach();
    });
}

bool Bank::is_remote(int64_t uid) const {
    // FIXME: make generic
    std::unique_lock guard(m_mutex);

    for (const auto& bank : m_device_banks) {
        if (ov::npuw::util::starts_with(bank.first, "NPU")) {
            if (bank.second.storage.find(uid) != bank.second.storage.end()) {
                // Found in NPU bank so considered remote (utterly wrong for the generic case)
                return true;
            }
        }
    }
    return false;
}

void Bank::serialize(ov::npuw::orc::Stream& stream) {
    LOG_INFO("Serializing weights bank...");
    LOG_BLOCK();

    std::unique_lock guard(m_mutex);

    std::size_t bank_size = m_device_banks.size();
    stream & bank_size;

    for (const auto& elem : m_device_banks) {
        const auto& device = elem.first;
        const auto& device_bank = elem.second;
        auto storage_size = device_bank.storage.size();
        stream & device & storage_size;
        // Write tensors sequentially according to sorted uids for better memory allocation and utilization
        std::set<int64_t> uids;
        for (const auto& t_pair : device_bank.storage) {
            uids.insert(t_pair.first);
        }

        for (const auto& uid : uids) {
            stream & uid;
            auto tensor = device_bank.storage.at(uid).tensor;
            transfer_tensor(stream, tensor);
        }
    }

    LOG_INFO("DONE.");
}

void Bank::read_and_add_tensor(ov::npuw::orc::Stream& stream, int64_t uid, const std::string& device) {
    // This method is supposed to be used only during deserialization
    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks[device];
    auto iter_device = device_bank.storage.find(uid);

    if (iter_device != device_bank.storage.end()) {
        // Shouldn't be possible
        NPUW_ASSERT(false);
        return;
    }

    if (device == "CPU") {
        // Just read deserialized tensor into the bank
        transfer_tensor(stream, device_bank.storage[uid].tensor);
        return;
    }

    // Need to allocate on device and copy deserialized tensor to that memory
    auto remote_ctx = m_core->get_default_context(device)._ptr;
    transfer_tensor(stream,
                    device_bank.storage[uid].tensor,
                    [&remote_ctx](const ov::element::Type& type, const ov::Shape& shape) {
                        ov::SoPtr<ov::ITensor> remote_tensor = remote_ctx->create_host_tensor(type, shape);
                        return ov::make_tensor(remote_tensor);
                    });
    NPUW_ASSERT(device_bank.storage[uid].tensor && "Remote tensor should be initialized during bank deserialize");
    device_bank.storage[uid].lt = LazyTensor();
}

std::string Bank::get_name() const {
    return m_bank_name;
}

void ov::npuw::orc::serialize(Stream& stream, ov::npuw::weights::Bank& var) {
    if (stream.output()) {
        var.serialize(stream);
    } else {
        LOG_INFO("Deserializing weights bank...");
        LOG_BLOCK();

        std::size_t bank_size = 0;
        stream & bank_size;

        for (std::size_t i = 0; i < bank_size; ++i) {
            std::string device;
            stream & device;
            std::size_t storage_size = 0;
            stream & storage_size;
            for (std::size_t j = 0; j < storage_size; ++j) {
                int64_t uid = -1;
                stream & uid;
                var.read_and_add_tensor(stream, uid, device);
            }
        }

        LOG_INFO("DONE.");
    }
}

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name,
                                           const std::shared_ptr<const ov::ICore>& core,
                                           const std::string& alloc_device,
                                           std::shared_ptr<ov::weight_sharing::Context> weight_shared_context) {
    std::unique_lock guard(m_mutex);

    auto iter = m_bank_map.find(bank_name);
    if (iter == m_bank_map.end() || iter->second.expired()) {
        auto bank = std::make_shared<Bank>(core, alloc_device, bank_name, weight_shared_context);
        m_bank_map[bank_name] = bank;
        return bank;
    }
    return iter->second.lock();
}

std::shared_ptr<Bank> ov::npuw::weights::bank(const std::string& bank_name,
                                              const std::shared_ptr<const ov::ICore>& core,
                                              const std::string& alloc_device,
                                              std::shared_ptr<ov::weight_sharing::Context> weight_shared_context) {
    if (bank_name.empty()) {
        // Don't share this bank in manager
        return std::make_shared<Bank>(core, alloc_device, bank_name, weight_shared_context);
    }

    auto& instance = BankManager::getInstance();
    return instance.getBank(bank_name, core, alloc_device, weight_shared_context);
}
