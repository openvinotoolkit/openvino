// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include <cstring>

#include "logging.hpp"
#include "openvino/core/parallel.hpp"
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
                                  const std::string& alloc_device);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

Bank::Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device, const std::string& bank_name)
    : m_core(core),
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

namespace {
// Reshaped view of an already allocated buffer, keeps the owner alive
ov::Tensor make_alias_view(const ov::Tensor& owner, const LazyTensor::Meta& meta) {
    auto holder = std::make_shared<ov::Tensor>(owner);
    // Non-const data(): a const pointer would make the view reject mutable access
    auto view = ov::make_tensor(meta.type, meta.shape, holder->data());
    NPUW_ASSERT(view->get_byte_size() == owner.get_byte_size());
    return ov::make_tensor(ov::SoPtr<ov::ITensor>(view, holder));
}

// A shorter view or another element type may start at the same address: only equal type and size can be shared
bool same_buffer_layout(const LazyTensor::Meta& a, const LazyTensor::Meta& b) {
    return a.type == b.type && ov::shape_size(a.shape) == ov::shape_size(b.shape);
}
}  // namespace

// The source buffer of an earlier owner may have been freed and its address reused by different data, so a hit is
// only taken after comparing the bytes: the candidate's own source is alive here, the owner's copy is host-accessible.
Bank::Views Bank::find_earlier_owners(const DeviceBank& device_bank, const Candidates& candidates) {
    struct Match {
        int64_t uid;
        int64_t owner;
        const void* source;
        bool same_bytes = false;
    };
    std::vector<Match> matches;
    for (const auto& [uid, lt] : candidates) {
        const void* source = lt.const_data_ptr();
        if (source == nullptr) {
            continue;
        }
        const auto iter_alias = device_bank.aliases.find(source);
        if (iter_alias == device_bank.aliases.end()) {
            continue;
        }
        const auto iter_owner = device_bank.storage.find(iter_alias->second);
        if (iter_owner == device_bank.storage.end() || !iter_owner->second.tensor) {
            continue;
        }
        const auto& owner = iter_owner->second.tensor;
        if (same_buffer_layout(lt.eval_meta(), {owner.get_shape(), owner.get_element_type()})) {
            matches.push_back({uid, iter_alias->second, source});
        }
    }
    ov::parallel_for(matches.size(), [&](std::size_t idx) {
        auto& match = matches[idx];
        const auto& owner = device_bank.storage.at(match.owner).tensor;
        match.same_bytes = std::memcmp(match.source, owner.data(), owner.get_byte_size()) == 0;
    });

    Views views;
    for (const auto& match : matches) {
        if (match.same_bytes) {
            views.emplace(match.uid, match.owner);
        }
    }
    return views;
}

// Within one pass every source buffer is alive, so equal pointers mean the same buffer: the lowest uid owns it and
// the rest become views of it. Returns the new owners.
Bank::BufferOwners Bank::split_owners_and_views(const Candidates& candidates, Views& views) {
    BufferOwners owners;
    for (const auto& [uid, lt] : candidates) {
        if (views.count(uid) != 0) {
            continue;
        }
        const void* source = lt.const_data_ptr();
        if (source == nullptr) {
            continue;
        }
        const auto [iter_owner, inserted] = owners.emplace(source, uid);
        if (!inserted && same_buffer_layout(lt.eval_meta(), candidates.at(iter_owner->second).eval_meta())) {
            views.emplace(uid, iter_owner->second);
        }
    }
    return owners;
}

void Bank::bind_views(DeviceBank& device_bank, const Views& views, const std::string& device) {
    std::size_t shared_bytes = 0u;
    for (const auto& [uid, owner_uid] : views) {
        const auto& owner = device_bank.storage.at(owner_uid).tensor;
        NPUW_ASSERT(owner && "Owner of a shared tensor should be allocated first!");
        auto& stored_tensor = device_bank.storage.at(uid);
        stored_tensor.tensor = make_alias_view(owner, stored_tensor.lt.eval_meta());
        shared_bytes += owner.get_byte_size();
        stored_tensor.lt.detach();
    }
    if (!views.empty()) {
        LOG_INFO("WeightsBank: shared " << views.size() << " tensors with existing allocations on " << device
                                        << ", saved " << shared_bytes / 1024.0 / 1024.0 << " MiB");
    }
}

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
        device_bank.storage.at(uid).tensor = ov::Tensor(t.get_element_type(), t.get_shape());
        // Get ownership of the weights, might be a mmaped object during import
        t.copy_to(device_bank.storage.at(uid).tensor);
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
    Candidates candidates;
    for (const auto& lt : to_process) {
        auto iter_device_registered = device_bank.registered_tensors.find(lt);
        NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                    "Tensor should be registered first!");
        candidates.emplace(iter_device_registered->second, lt);
    }

    // Tensors which only differ by shape from an already allocated one are views of it, not new allocations
    auto views = find_earlier_owners(device_bank, candidates);
    const auto owners = split_owners_and_views(candidates, views);

    std::vector<TensorToAllocate> uids_to_allocated;
    uids_to_allocated.reserve(candidates.size() - views.size());
    for (const auto& [uid, lt] : candidates) {
        if (views.count(uid) == 0) {
            uids_to_allocated.push_back({lt.eval_meta(), ov::Tensor(), uid});
        }
    }

    // Allocate memory sequentially - in order of UID
    auto remote_ctx = m_core->get_default_context(device)._ptr;
    for (auto&& allocated : uids_to_allocated) {
        ov::SoPtr<ov::ITensor> remote_tensor =
            remote_ctx->create_host_tensor(allocated.meta.type, allocated.meta.shape);
        allocated.allocated_tensor = ov::make_tensor(remote_tensor);
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

    // Published only once the owners are allocated, so a failed pass leaves no entry behind
    for (const auto& [key, uid] : owners) {
        device_bank.aliases[key] = uid;
    }
    bind_views(device_bank, views, device);
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
                                           const std::string& alloc_device) {
    std::unique_lock guard(m_mutex);

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
