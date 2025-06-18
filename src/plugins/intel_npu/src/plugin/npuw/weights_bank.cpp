// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "serialization.hpp"
#include "util.hpp"

using ov::npuw::weights::Bank;
using ov::npuw::weights::LazyTensor;
using ov::npuw::weights::RemoteContextManager;

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
                                  const std::string& alloc_device);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

class RemoteContextManager {
public:
    ov::SoPtr<ov::IRemoteContext> getContext(const std::shared_ptr<const ov::ICore>& core, const std::string& device) {
        auto it_ctx = m_context_map.find(device);
        if (it_ctx == m_context_map.end()) {
            m_context_map[device] = core->get_default_context(device)._ptr;
            it_ctx = m_context_map.find(device);
        }
        return it_ctx->second;
    }

private:
    std::unordered_map<std::string, ov::SoPtr<ov::IRemoteContext>> m_context_map;
};

Bank::Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device, const std::string& bank_name)
    : m_core(core),
      m_alloc_device(alloc_device),
      m_bank_name(bank_name),
      m_rcm(std::make_shared<RemoteContextManager>()) {}

int64_t Bank::registerLT(const LazyTensor& tensor, const std::string& device) {
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    std::lock_guard<std::mutex> guard(m_mutex);

    auto& device_bank = m_device_banks[device_for_alloc];
    std::unique_lock dev_guard(device_bank.mutex);

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

    std::lock_guard<std::mutex> guard(m_mutex);

    auto& device_bank = m_device_banks[device_for_alloc];

    std::unique_lock<std::mutex> dev_guard(device_bank.mutex);
    auto iter_device = device_bank.storage.find(uid);

    NPUW_ASSERT(iter_device != device_bank.storage.end() && iter_device->second.tensor &&
                "Tensor should be registered and allocated first!");

    return iter_device->second.tensor;
}

struct AllocatedTensor {
    bool needs_alloc = false;
    LazyTensor::Meta meta;
    ov::Tensor allocated_tensor;
};

void Bank::evaluate_and_allocate() {
    std::lock_guard<std::mutex> guard(m_mutex);

    for (auto&& bank : m_device_banks) {
        const auto& device_for_alloc = bank.first;
        auto& device_bank = bank.second;

        std::vector<LazyTensor> vec;

        std::unique_lock storage_guard(device_bank.mutex);
        vec.reserve(device_bank.storage.size());
        for (const auto& el : device_bank.storage) {
            // Add non-allocated tensors for furter evaluation and allocation
            if (!el.second.tensor) {
                vec.push_back(el.second.lt);
            }
        }
        storage_guard.unlock();

        std::vector<AllocatedTensor> uids_to_allocated(uid_count);

        // No allocation needed
        if (device_for_alloc == "CPU") {
            ov::parallel_for(vec.size(), [&](std::size_t idx) {
                const auto& lt = vec[idx];
                std::unique_lock dev_guard(device_bank.mutex);
                auto iter_device_registered = device_bank.registered_tensors.find(lt);
                NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                            "Tensor should be registered first!");
                auto uid = iter_device_registered->second;
                if (device_bank.storage[uid].tensor) {
                    // Already allocated
                    return;
                }

                dev_guard.unlock();
                auto transformed = lt.eval();

                // Note: No need to relock here as the storage is already prepared for writing
                device_bank.storage.at(uid).tensor = std::move(transformed);
                const_cast<LazyTensor&>(lt).detach();
            });
        } else {
            // Allocation needed
            std::unique_lock dev_guard(device_bank.mutex);
            for (std::size_t idx = 0; idx < vec.size(); ++idx) {
                const auto& lt = vec[idx];
                auto iter_device_registered = device_bank.registered_tensors.find(lt);
                NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                            "Tensor should be registered first!");
                auto uid = iter_device_registered->second;
                if (device_bank.storage[uid].tensor) {
                    // Already allocated
                    continue;
                }

                uids_to_allocated[uid] = {true, lt.eval_meta(), ov::Tensor()};
            }
        }

        std::unique_lock guard(device_bank.mutex);
        // Allocate memory sequentially - in order of UID
        for (std::size_t i = 0; i < uids_to_allocated.size(); ++i) {
            auto& allocated = uids_to_allocated[i];
            if (!allocated.needs_alloc) {
                continue;
            }
            auto uid = i;

            ov::SoPtr<ov::ITensor> remote_tensor;

            remote_tensor =
                get_context(m_core, device_for_alloc)->create_host_tensor(allocated.meta.type, allocated.meta.shape);
            uids_to_allocated.at(uid) = {true, allocated.meta, ov::make_tensor(remote_tensor)};
        }
        guard.unlock();

        // Copy to allocated memory
        // Evaluate concurrently, lock the device
        // mutex only to update the device bank (& allocate on-device memory, if needed)
        ov::parallel_for(uids_to_allocated.size(), [&](std::size_t idx) {
            std::unique_lock dev_guard(device_bank.mutex);
            auto& allocated = uids_to_allocated[idx];
            if (!allocated.needs_alloc) {
                return;
            }
            auto uid = idx;
            dev_guard.unlock();

            auto transformed = device_bank.storage.at(uid).lt.eval();
            transformed.copy_to(allocated.allocated_tensor);

            // Note: no need to relock the mutex here as there should be only 1 writer to that specific index
            device_bank.storage.at(uid).tensor = std::move(allocated.allocated_tensor);

            // Detach the evaluated LazyTensor from its memory here - when it is 100%
            // not needed anymore (transformations, if any, and copies are done)
            // Note: this is the non-CPU path!
            const_cast<LazyTensor&>(device_bank.storage.at(uid).lt).detach();
        });
    }
}

bool Bank::is_remote(int64_t uid) const {
    // FIXME: make generic
    std::lock_guard<std::mutex> guard(m_mutex);

    auto npu_bank = m_device_banks.find("NPU");
    if (npu_bank != m_device_banks.end()) {
        std::lock_guard<std::mutex> dev_guard(npu_bank->second.mutex);
        if (npu_bank->second.storage.find(uid) != npu_bank->second.storage.end()) {
            // Found in NPU bank so considered remote (utterly wrong for the generic case)
            return true;
        }
    }
    return false;
}

void Bank::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;

    LOG_INFO("Serializing weights bank...");
    LOG_BLOCK();

    std::lock_guard<std::mutex> guard(m_mutex);

    write(stream, m_device_banks.size());

    for (const auto& elem : m_device_banks) {
        const auto& device = elem.first;
        const auto& device_bank = elem.second;
        std::lock_guard<std::mutex> dev_guard(device_bank.mutex);
        write(stream, device);
        write(stream, device_bank.storage.size());
        // Write tensors sequentially according to sorted uids for better memory allocation and utilization
        std::set<int64_t> uids;
        for (const auto& t_pair : device_bank.storage) {
            uids.insert(t_pair.first);
        }

        for (const auto& uid : uids) {
            write(stream, uid);
            write(stream, device_bank.storage.at(uid).tensor);
        }
    }

    LOG_INFO("DONE.");
}

std::shared_ptr<Bank> Bank::deserialize(std::istream& stream,
                                        const std::shared_ptr<const ov::ICore>& core,
                                        const std::string& name) {
    using namespace ov::npuw::s11n;

    LOG_INFO("Deserializing weights bank...");
    LOG_BLOCK();

    auto bank = ov::npuw::weights::bank(name, core, "");

    std::size_t bank_size = 0;
    read(stream, bank_size);

    for (std::size_t i = 0; i < bank_size; ++i) {
        std::string device;
        read(stream, device);
        std::size_t storage_size = 0;
        read(stream, storage_size);
        for (std::size_t j = 0; j < storage_size; ++j) {
            int64_t uid = -1;
            read(stream, uid);
            bank->read_and_add_tensor(stream, uid, device);
        }
    }

    LOG_INFO("DONE.");

    return bank;
}

void Bank::read_and_add_tensor(std::istream& stream, int64_t uid, const std::string& device) {
    using namespace ov::npuw::s11n;

    // This method is supposed to be used only during deserialization
    std::lock_guard<std::mutex> guard(m_mutex);

    auto& device_bank = m_device_banks[device];
    std::lock_guard<std::mutex> dev_guard(device_bank.mutex);

    auto iter_device = device_bank.storage.find(uid);

    if (iter_device != device_bank.storage.end()) {
        // Shouldn't be possible
        NPUW_ASSERT(false);
        return;
    }

    if (device == "CPU") {
        // Just read deserialized tensor into the bank
        read(stream, device_bank.storage[uid].tensor);
        return;
    }

    // Need to allocate on device and copy deserialized tensor to that memory
    ov::SoPtr<ov::ITensor> remote_tensor;
    ov::Tensor allocated_tensor;

    // FIXME: reading not via a dedicated function
    bool is_intialized = false;
    read(stream, is_intialized);
    NPUW_ASSERT(is_intialized);

    std::string type_str;
    read(stream, type_str);
    ov::element::Type type(type_str);

    ov::Shape shape;
    read(stream, shape);

    std::size_t byte_size = 0;
    read(stream, byte_size);

    remote_tensor = get_context(m_core, device)->create_host_tensor(type, shape);
    allocated_tensor = ov::make_tensor(remote_tensor);
    device_bank.storage[uid] = {LazyTensor(), allocated_tensor};
    stream.read(reinterpret_cast<char*>(allocated_tensor.data()), byte_size);
}

std::string Bank::get_name() const {
    return m_bank_name;
}

ov::SoPtr<ov::IRemoteContext> Bank::get_context(const std::shared_ptr<const ov::ICore>& core,
                                                const std::string& device) {
    // Note: don't lock Bank's mutexes here as this function should be already called under a locked mutex
    return m_rcm->getContext(core, device);
}

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name,
                                           const std::shared_ptr<const ov::ICore>& core,
                                           const std::string& alloc_device) {
    std::lock_guard<std::mutex> guard(m_mutex);

    auto iter = m_bank_map.find(bank_name);
    if (iter == m_bank_map.end() || iter->second.expired()) {
        auto bank = std::make_shared<Bank>(core, alloc_device, bank_name);
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
        return std::make_shared<Bank>(core, alloc_device, bank_name);
    }

    auto& instance = BankManager::getInstance();
    return instance.getBank(bank_name, core, alloc_device);
}
