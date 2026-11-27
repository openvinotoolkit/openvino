// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "intel_npu/utils/stable_tensor_view.hpp"
#include "logging.hpp"
#include "openvino/core/memory_util.hpp"
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
    return register_tensor(device_bank, tensor, device_for_alloc != "CPU");
}

int64_t Bank::register_tensor(DeviceBank& device_bank, const LazyTensor& tensor, bool use_reshape_views) {
    std::optional<int64_t> owner_uid;
    if (use_reshape_views) {
        const auto source = tensor.reshape_source();
        if (source) {
            owner_uid = register_tensor(device_bank, source, true);
        }
    }

    const void* source = nullptr;
    ov::Tensor imported;
    bool source_initialized = false;
    const auto range = device_bank.registered_tensors.equal_range(tensor);
    for (auto iter = range.first; iter != range.second; ++iter) {
        const auto& stored = device_bank.storage.at(iter->second);
        if (owner_uid != stored.owner_uid) {
            continue;
        }
        if (!source_initialized) {
            source = tensor.const_data_ptr();
            if (!source) {
                imported = tensor.imported_const_for_comparison();
                if (imported) {
                    source = imported.data();
                }
            }
            source_initialized = true;
        }
        if (!source && tensor.is_plain_constant() && !tensor.shares_impl_with(stored.lt)) {
            continue;
        }
        const void* previous = nullptr;
        ov::Tensor previous_imported;
        if (stored.tensor) {
            previous = stored.tensor.data();
        } else {
            previous = stored.lt.const_data_ptr();
            if (!previous) {
                previous_imported = stored.lt.imported_const_for_comparison();
                if (previous_imported) {
                    previous = previous_imported.data();
                }
            }
        }
        if (source && !previous) {
            continue;
        }
        if (source && previous && source != previous) {
            const auto meta = tensor.eval_meta();
            const auto size = ov::util::get_memory_size_safe(meta.type, meta.shape);
            OPENVINO_ASSERT(size, "Weights bank tensor size overflows");
            if (std::memcmp(source, previous, *size) != 0) {
                continue;
            }
        }
        // A duplicate may share its source with another tensor in this pass.
        device_bank.duplicates.push_back(tensor);
        return iter->second;
    }
    NPUW_ASSERT(uid_count < std::numeric_limits<int64_t>::max() && "Weights bank uid space exhausted");
    const auto uid = uid_count++;
    device_bank.registered_tensors.emplace(tensor, uid);
    device_bank.storage[uid] = {tensor, ov::Tensor(), owner_uid};
    return uid;
}

ov::Tensor Bank::get(int64_t uid, const std::string& device) {
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    std::unique_lock guard(m_mutex);

    auto iter_bank =
        std::find_if(m_device_banks.begin(), m_device_banks.end(), [&device_for_alloc, uid](const auto& device_bank) {
            if (device_bank.second.storage.count(uid) != 1) {
                return false;
            }
            // Exact device key, or the un-indexed "NPU" key written by old blobs,
            // which serves any NPU.<n> request for the uids restored under it. uids are unique across
            // the bank, so at most one key can hold a given uid.
            return device_bank.first == device_for_alloc ||
                   (device_bank.first == "NPU" && ov::npuw::util::starts_with(device_for_alloc, "NPU."));
        });
    OPENVINO_ASSERT(iter_bank != m_device_banks.end(),
                    "NPUW: weights bank '",
                    m_bank_name,
                    "' has no storage for device '",
                    device_for_alloc,
                    "'");
    auto& device_bank = iter_bank->second;
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
    NPUW_ASSERT(owner.get_element_type() == meta.type);
    auto owner_impl = ov::get_tensor_impl(owner);
    auto view = std::make_shared<intel_npu::StableTensorView>(owner_impl, meta.shape);
    return ov::make_tensor(ov::SoPtr<ov::ITensor>(view, owner_impl._so));
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
        for (const auto owner_uid : iter_alias->second) {
            const auto iter_owner = device_bank.storage.find(owner_uid);
            if (iter_owner == device_bank.storage.end() || !iter_owner->second.tensor) {
                continue;
            }
            const auto& owner = iter_owner->second.tensor;
            if (same_buffer_layout(lt.eval_meta(), {owner.get_shape(), owner.get_element_type()})) {
                matches.push_back({uid, owner_uid, source});
                break;
            }
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
        auto& source_owners = owners[source];
        const auto meta = lt.eval_meta();
        const auto iter_owner = std::find_if(source_owners.begin(), source_owners.end(), [&](int64_t owner_uid) {
            return same_buffer_layout(meta, candidates.at(owner_uid).eval_meta());
        });
        if (iter_owner == source_owners.end()) {
            source_owners.push_back(uid);
        } else {
            views.emplace(uid, *iter_owner);
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
    }
    if (!views.empty()) {
        LOG_INFO("WeightsBank: shared " << views.size() << " tensors with existing allocations on " << device
                                        << ", covering " << shared_bytes / 1024.0 / 1024.0 << " MiB of views");
    }
}

void Bank::evaluate_and_allocate() {
    std::unique_lock guard(m_mutex);
    std::vector<LazyTensor> to_detach;

    for (auto&& bank : m_device_banks) {
        const auto& device_for_alloc = bank.first;
        auto& device_bank = bank.second;

        Pending to_process;
        to_process.reserve(device_bank.storage.size());
        for (const auto& el : device_bank.storage) {
            // Add non-allocated tensors for furter evaluation and allocation
            if (!el.second.tensor) {
                to_process.emplace_back(el.first, el.second.lt);
                to_detach.push_back(el.second.lt);
            }
        }

        if (device_for_alloc == "CPU") {
            evaluate_cpu(device_bank, to_process);
        } else {
            evaluate_and_allocate_on_device(device_bank, to_process, device_for_alloc);
        }
    }  // for (m_device_banks)
    for (auto& tensor : to_detach) {
        tensor.detach();
    }
    for (auto& bank : m_device_banks) {
        for (auto& duplicate : bank.second.duplicates) {
            duplicate.detach();
        }
        bank.second.duplicates.clear();
    }
}

void Bank::evaluate_cpu(Bank::DeviceBank& device_bank, const Pending& to_process) {
    // Note: not locking here. This is a private function, so Bank should handle the locks around it
    // as we lock in evaluate_and_allocate() now.
    ov::parallel_for(to_process.size(), [&](std::size_t idx) {
        const auto& [uid, lt] = to_process[idx];
        auto t = lt.eval();
        device_bank.storage.at(uid).tensor = ov::Tensor(t.get_element_type(), t.get_shape());
        // Get ownership of the weights, might be a mmaped object during import
        t.copy_to(device_bank.storage.at(uid).tensor);
    });
}

// Note: there are no locks in this function's parallel_for
// At this point all the LazyTensor->Tensor pairs in the map are already
// allocated and there are no conflicting reads/writes since all the tensors are unique.

// FIXME: this whole flow could be improved for the same bank
// processing from different threads. We could separate LazyTensors
// evaluation and the bank access. But it requires additional rework.
void Bank::evaluate_and_allocate_on_device(Bank::DeviceBank& device_bank,
                                           const Pending& to_process,
                                           const std::string& device) {
    // Note: not locking here. This is a private function, so Bank should handle the locks around it
    // as we lock in evaluate_and_allocate() now.
    Candidates candidates;
    for (const auto& [uid, lt] : to_process) {
        candidates.emplace(uid, lt);
    }

    // Tensors which only differ by shape from an already allocated one are views of it, not new allocations
    auto views = find_earlier_owners(device_bank, candidates);
    for (const auto& [uid, lt] : candidates) {
        const auto& owner_uid = device_bank.storage.at(uid).owner_uid;
        if (owner_uid) {
            views.emplace(uid, *owner_uid);
        }
    }
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
    });

    // Published only once the owners are allocated, so a failed pass leaves no entry behind
    for (const auto& [key, new_owners] : owners) {
        auto& stored_owners = device_bank.aliases[key];
        for (const auto uid : new_owners) {
            const auto& new_owner = device_bank.storage.at(uid).tensor;
            const auto iter = std::find_if(stored_owners.begin(), stored_owners.end(), [&](int64_t owner_uid) {
                const auto& owner = device_bank.storage.at(owner_uid).tensor;
                return same_buffer_layout({new_owner.get_shape(), new_owner.get_element_type()},
                                          {owner.get_shape(), owner.get_element_type()});
            });
            if (iter == stored_owners.end()) {
                stored_owners.push_back(uid);
            } else {
                *iter = uid;
            }
        }
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

void Bank::serialize(ov::npuw::orc::Stream& stream, bool preserve_views) {
    LOG_INFO("Serializing weights bank...");
    LOG_BLOCK();

    std::unique_lock guard(m_mutex);

    if (preserve_views) {
        // The reserved bank count distinguishes this payload from the frozen v0 layout.
        std::size_t marker = std::numeric_limits<std::size_t>::max();
        std::uint16_t version = 1;
        stream & marker & version;
    }
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

        BufferOwners serialized_owners;
        for (const auto& uid : uids) {
            stream & uid;
            auto tensor = device_bank.storage.at(uid).tensor;
            int64_t owner_uid = -1;
            if (preserve_views && device != "CPU" && tensor && tensor.get_byte_size() != 0) {
                auto& owners = serialized_owners[tensor.data()];
                for (const auto candidate_uid : owners) {
                    const auto& candidate = device_bank.storage.at(candidate_uid).tensor;
                    if (same_buffer_layout({tensor.get_shape(), tensor.get_element_type()},
                                           {candidate.get_shape(), candidate.get_element_type()})) {
                        owner_uid = candidate_uid;
                        break;
                    }
                }
                if (owner_uid == -1) {
                    owners.push_back(uid);
                }
            }
            if (preserve_views) {
                const bool is_view = owner_uid != -1;
                stream & is_view;
                if (is_view) {
                    auto shape = tensor.get_shape();
                    stream & owner_uid & shape;
                    continue;
                }
            }
            transfer_tensor(stream, tensor);
        }
    }

    LOG_INFO("DONE.");
}

void Bank::read_and_add_view(int64_t uid, int64_t owner_uid, const ov::Shape& shape, const std::string& device) {
    std::unique_lock guard(m_mutex);
    NPUW_ASSERT(uid >= 0 && uid < std::numeric_limits<int64_t>::max() && "Malformed blob: invalid weights bank uid");
    NPUW_ASSERT(owner_uid >= 0 && owner_uid < uid && "Malformed blob: invalid weights bank view owner");
    for (const auto& bank : m_device_banks) {
        NPUW_ASSERT(bank.second.storage.count(uid) == 0 && "Malformed blob: duplicate weights bank uid");
    }
    auto& device_bank = m_device_banks[device];
    const auto owner_it = device_bank.storage.find(owner_uid);
    NPUW_ASSERT(owner_it != device_bank.storage.end() && owner_it->second.tensor &&
                "Malformed blob: weights bank view owner not found");
    const auto& owner = owner_it->second.tensor;
    const auto byte_size = ov::util::get_memory_size_safe(owner.get_element_type(), shape);
    NPUW_ASSERT(byte_size && *byte_size == owner.get_byte_size() && ov::shape_size(shape) == owner.get_size() &&
                "Malformed blob: weights bank view shape does not match owner");
    device_bank.storage[uid].tensor = make_alias_view(owner, {shape, owner.get_element_type()});
    device_bank.storage[uid].owner_uid = owner_uid;
    uid_count = std::max(uid_count, uid + 1);
}

void Bank::read_and_add_tensor(ov::npuw::orc::Stream& stream, int64_t uid, const std::string& device) {
    // This method is supposed to be used only during deserialization
    std::unique_lock guard(m_mutex);
    NPUW_ASSERT(uid >= 0 && "Malformed blob: negative weights bank uid");
    NPUW_ASSERT(static_cast<uint64_t>(uid) < static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                "Malformed blob: weights bank uid exceeds max int value");

    for (const auto& bank : m_device_banks) {
        NPUW_ASSERT(bank.second.storage.count(uid) == 0 && "Malformed blob: duplicate weights bank uid");
    }
    auto& device_bank = m_device_banks[device];
    uid_count = std::max(uid_count, uid + 1);

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

void Bank::deserialize(ov::npuw::orc::Stream& stream, std::optional<ov::npuw::orc::Version> expected_version) {
    LOG_INFO("Deserializing weights bank...");
    LOG_BLOCK();

    std::size_t bank_size = 0;
    stream & bank_size;
    bool preserve_views = false;
    if (bank_size == std::numeric_limits<std::size_t>::max()) {
        std::uint16_t version = 0;
        stream & version;
        OPENVINO_ASSERT(version == 1, "Unsupported weights bank payload version ", version);
        stream & bank_size;
        preserve_views = true;
    }
    OPENVINO_ASSERT(!expected_version || *expected_version == (preserve_views ? 1u : 0u),
                    "Weights bank section and payload versions do not match");

    for (std::size_t i = 0; i < bank_size; ++i) {
        std::string device;
        stream & device;
        std::size_t storage_size = 0;
        stream & storage_size;
        for (std::size_t j = 0; j < storage_size; ++j) {
            int64_t uid = -1;
            stream & uid;
            bool is_view = false;
            if (preserve_views) {
                stream & is_view;
            }
            if (is_view) {
                int64_t owner_uid = -1;
                ov::Shape shape;
                stream & owner_uid & shape;
                read_and_add_view(uid, owner_uid, shape, device);
            } else {
                read_and_add_tensor(stream, uid, device);
            }
        }
    }
    LOG_INFO("DONE.");
}

void ov::npuw::orc::serialize(Stream& stream, ov::npuw::weights::Bank& var) {
    if (stream.output()) {
        var.serialize(stream);
    } else {
        var.deserialize(stream);
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
