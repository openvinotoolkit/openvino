// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include <map>

#include "logging.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "serialization.hpp"
#include "util.hpp"

using ov::npuw::weights::Bank;
using ov::npuw::weights::LazyTensor;

namespace {
ov::AnyMap prepareNPURemoteMemoryImportParams(std::shared_ptr<ov::AlignedBuffer> buffer_to_import) {
    if (!buffer_to_import)  {
        return {};
    }
    ov::AnyMap params {
        ov::intel_npu::mem_type(ov::intel_npu::MemType::CPU_VA),
        ov::intel_npu::mem_handle(buffer_to_import->get_ptr()),
        ov::intel_npu::tensor_type(ov::intel_npu::TensorType::BINDED),
    };
    return params;
}

ov::Tensor tensor_factory(const ov::ICore &core, const std::string &device,
                          const ov::element::Type& type, const ov::Shape& shape,
                          std::shared_ptr<ov::AlignedBuffer> staging_buffer_ptr = {}) {
    if (device == "CPU") {
        return staging_buffer_ptr ? ov::Tensor(type, shape, staging_buffer_ptr->get_ptr<char>()) :
                                    ov::Tensor(type, shape);
    }

    auto remote_ctx = core.get_default_context(device)._ptr;
    ov::AnyMap params;
    if (device == "NPU") {
        params = prepareNPURemoteMemoryImportParams(staging_buffer_ptr);
    } else if (device == "GPU") {
        NPUW_ASSERT(false && "create_tensor_allocator: GPU remote context parameter creator is not implemented");
    }

    NPUW_ASSERT(remote_ctx && "create_tensor_allocator:Remote context is not initialized");
    auto staging_buffer_ptr_copy = staging_buffer_ptr;
    return !params.empty() ? ov::make_tensor(remote_ctx->create_tensor(type, shape, params)) :
                             ov::make_tensor(remote_ctx->create_host_tensor(type, shape));
}
}

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

const ov::op::v0::Constant& ov::npuw::weights::WeightSharingContextAccessor::get_transformed_constant(
    const std::string& device,
    size_t source_id,
    Handle& handle) const {
    OPENVINO_ASSERT(handle.context, "WeightSharingContextAccessor has no weight sharing context");
    auto dev_it = m_transformed_weights_per_device.find(device);
    OPENVINO_ASSERT(dev_it != m_transformed_weights_per_device.end(), "No transformed weights found for device: ", device);
    auto iter = dev_it->second.find(source_id);
    OPENVINO_ASSERT(iter != dev_it->second.end(), "Transformed constant not found for source id: ", source_id, ", on device: ", device);
    return iter->second;
}

ov::weight_sharing::WeightBuffer ov::npuw::weights::WeightSharingContextAccessor::request_shared_weight_source_buffer(
    size_t source_id,
    const element::Type& type, const ov::Shape& shape,
    const std::string& device,
    Handle& handle) {
    if (!handle.context) {
        return {};
    }

    auto size = ov::util::get_memory_size_safe(type, shape);
    OPENVINO_ASSERT(size.has_value(), "LazyTensor size overflow, uid: ", source_id, " device: ", device);
    LOG_DEBUG("WeightSharingContextAccessor::request_shared_weight_source_buffer - source id: " << source_id << ", device: " << device << ", size: " << *size);

    auto shared_weight_source = ov::weight_sharing::get_runtime_weight_source(*handle.context, source_id);
    if (auto shared_buffer = shared_weight_source.m_weights.lock(); shared_buffer != nullptr) {
        OPENVINO_ASSERT(shared_weight_source.m_device == device,
                        "Shared weight source exists for the wrong device, expected: ",
                        device,
                        " actual: ",
                        shared_weight_source.m_device);
        LOG_VERB("WeightSharingContextAccessor::request_shared_weight_source_buffer - reusing shared buffer for source id: " << source_id << ", device: " << device);
        return shared_buffer;
    }

    auto shared_buffer = WeightSharingContextAccessor::create_shared_buffer(source_id, *size);
    register_shared_weight_source_for_weight(shared_buffer, device, handle);

    // A transformed constant is never exist in the model as a node.
    // let's create a fake constant using the shared buffer as its data source
    // and register it with the weight sharing context.
    auto transformed_constant = ov::op::v0::Constant(type, shape, shared_buffer);
    ov::weight_sharing::set_constant(*handle.context, transformed_constant);

    // CVS-198039 WeightMetaData has no m_shape
    // As weight sharing context doesn't include the shape of the constant in WeightMetaData,
    // we store and remember the transformed constant,
    // so that we can deserialize its ov::Tensor from raw buffer.
    m_transformed_weights_per_device[device].erase(source_id);
    m_transformed_weights_per_device[device].emplace(source_id, std::move(transformed_constant));
    LOG_DEBUG("WeightSharingContextAccessor::request_shared_weight_source_buffer - m_transformed_weights_per_device: " << device << ", counts: " << m_transformed_weights_per_device[device].size());
    return shared_buffer;
}

bool ov::npuw::weights::WeightSharingContextAccessor::serialize_native_format(ov::npuw::orc::Stream& stream) {
    auto lock = get_handle();
    return serialize_native_format_impl(stream, lock);
}

bool ov::npuw::weights::WeightSharingContextAccessor::serialize_native_format_impl(ov::npuw::orc::Stream& stream, Handle& handle) const {
    using namespace ov::weight_sharing;
    NPUW_ASSERT(stream.output() && "WeightSharingContextAccessor::serialize_native_format must receive an output stream");
    if (!m_context) {
        return false;
    }
    // NPUW serializes the device at first then stores tensor UID within each device section by the incremental order.
    // Let's sort the source_id by device.
    std::map<std::string, std::set<DataID>> weight_sources_per_device;
    // for each source_id we collect constant ids associated with it
    std::map<DataID, std::set<DataID>> constant_ids_per_source_id_ordered;
    for (const auto& elem : m_context->m_runtime_sources) {
        const auto& [source_id, source] = elem;
        weight_sources_per_device[source.m_device].insert(source_id);

        auto weigt_reg_it = m_context->m_weight_registry.find(source_id);
        NPUW_ASSERT(weigt_reg_it != m_context->m_weight_registry.end() && "NPUW: Weight not found in registry");
        const WeightMetaMap& weight_meta_map = weigt_reg_it->second;
        for (const auto [constant_id, weight_meta] : weight_meta_map) {
            constant_ids_per_source_id_ordered[source_id].insert(constant_id);
        }
    }
    // serialization
    std::size_t bank_size = weight_sources_per_device.size();
    LOG_DEBUG("WeightSharingContextAccessor::serialize_native_format - total devices: " << bank_size);
    stream & bank_size;
    for (const auto& [device, source_ids]: weight_sources_per_device) {
        // 1) serialize the name of the device
        stream & device;

        // 2) serialize the number of UIDs for this device
        size_t num_weights = 0;
        num_weights = std::accumulate(constant_ids_per_source_id_ordered.begin(),
                                      constant_ids_per_source_id_ordered.end(), num_weights,
                                      [&source_ids](size_t sum, const auto& pair) {
                                          return source_ids.count(pair.first) ? sum + pair.second.size() : sum;
                                      });
        LOG_DEBUG("WeightSharingContextAccessor::serialize_native_format - total weights for device " << device << ": " << num_weights);
        stream & num_weights;

        // serialize weights in the incrementing order
        for (const auto& [source_id, constant_ids] : constant_ids_per_source_id_ordered) {
            if (!source_ids.count(source_id)) continue;

            // for NPUW we must not serialize constant_ids, which number is always 1,
            // and the only element has constant_id equals to 0 for each weight transformation
            NPUW_ASSERT(constant_ids.size() == 1 && "NPUW: transformed constant ids for each source must have exactly one constant id");
            for (auto constant_id : constant_ids) {
                NPUW_ASSERT(constant_id == 0 && "NPUW: constant id must be 0 for each weight transformation");
                // 3) serialize the weight identifier
                if (constant_id != invalid_constant_id) {
                    // Should newer happen
                    stream & constant_id;
                } else {
                    // the NPUW format: source_id is a real identifier for the weight
                    // when constant_id is always 0
                    stream & source_id;
                }
                const WeightMetaData& weight_meta_map = m_context->m_weight_registry.find(source_id)->second.find(constant_id)->second;
                bool is_initialized = true; // once the weight is found in m_weight_registry, it is considered initialized
                stream & is_initialized;

                // 4) serialize the weight metadata
                auto type_str = weight_meta_map.m_type.to_string();
                /*CVS-198039 WeightMetaData has no m_shape*/
                auto weight_map_it = m_transformed_weights_per_device.find(device);
                OPENVINO_ASSERT(weight_map_it != m_transformed_weights_per_device.end() && "WeightSharingContextAccessor: device: ", device, " not found in transformed weights map");
                auto shape_it = weight_map_it->second.find(source_id);
                OPENVINO_ASSERT(shape_it != weight_map_it->second.end() && "WeightSharingContextAccessor: source id: ", source_id, " not found in transformed weights map for device: ", device);
                auto shape = shape_it->second.get_shape();
                auto byte_size = weight_meta_map.m_size;
                stream & type_str & shape & byte_size;

                // 5) finally serialize the actual weight data
                const WeightSource &data_source = m_context->m_runtime_sources.find(source_id)->second;
                auto source_staging_buffer = data_source.m_weights.lock();
                OPENVINO_ASSERT(source_staging_buffer && "WeightSharingContextAccessor: staging buffer for source id: ", source_id, " must exist");
                auto source_staging_buffer_descriptor = source_staging_buffer->get_descriptor();
                OPENVINO_ASSERT(source_staging_buffer_descriptor && "WeightSharingContextAccessor: staging buffer for source id: ", source_id, " must have a valid descriptor");
                LOG_VERB("WeightSharingContextAccessor::serialize_native_format - serializing weight data for uid: " << source_id << ", source_id: " << source_staging_buffer_descriptor->get_id() << ", size: " << source_staging_buffer->size());
                stream.bytes(source_staging_buffer->get_ptr(weight_meta_map.m_offset), weight_meta_map.m_size);
            }
        }
    }
    return true;
}


bool ov::npuw::weights::WeightSharingContextAccessor::deserialize_native_format(ov::npuw::orc::Stream& stream, const ov::ICore &core) {
    auto lock = get_handle();
    return deserialize_native_format_impl(stream, core, lock);
}

bool ov::npuw::weights::WeightSharingContextAccessor::deserialize_native_format_impl(ov::npuw::orc::Stream& stream, const ov::ICore &core, Handle& handle) {
    NPUW_ASSERT(stream.input() && "WeightSharingContextAccessor::deserialize_native_format_impl must receive an input stream");

    // TODO who will create the m_context if it is not already created?
    if (!handle.context) {
        return false;
    }

    size_t bank_size = 0;
    stream & bank_size;
    LOG_DEBUG("WeightSharingContextAccessor::deserialize_native_format - total devices: " << bank_size);
    for (size_t i = 0; i < bank_size; ++i) {
        std::string device;
        stream & device;
        std::size_t storage_size = 0;
        stream & storage_size;
        LOG_DEBUG("WeightSharingContextAccessor::deserialize_native_format - total weights for device " << device << ": " << storage_size);
        for (std::size_t j = 0; j < storage_size; ++j) {
            int64_t uid = -1;
            stream & uid;

            auto common_allocator = [this, &core, &device, uid, &handle] (const ov::element::Type& type, const ov::Shape& shape) {
                auto shared_source = this->request_shared_weight_source_buffer(uid, type, shape, device, handle);
                // return reconstructed ov::Tensor
                return tensor_factory(core, device, type, shape, shared_source);
            };

            ov::Tensor regained_tensor;
            transfer_tensor(stream, regained_tensor, common_allocator);
            (void)regained_tensor; // can be dropped, not needed
            LOG_DEBUG("WeightSharingContextAccessor::deserialize_native_format - weight data restored, uid: " << uid);
        }
    }
    return true;
}

void ov::npuw::weights::WeightSharingContextAccessor::register_shared_weight_source_for_weight(
    ov::weight_sharing::WeightBuffer source_buffer,
    const std::string& device,
    Handle& handle) {

    auto buf_descriptor = source_buffer->get_descriptor();
    OPENVINO_ASSERT(buf_descriptor != nullptr, "WeightSharingContextAccessor::register_shared_weight_source_for_weight - must register a source buffer, that have a valid descriptor");
    size_t source_id = buf_descriptor->get_id();
    m_shared_sources[source_id] = source_buffer;
    ov::weight_sharing::set_runtime_weight_source(*handle.context, source_buffer, device);
    LOG_DEBUG("WeightSharingContextAccessor::register_shared_weight_source_for_weight - source id: " << source_id << ", device: " << device);
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
            if (!el.second.is_evaluated()) {
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
        auto shared_source = this->request_shared_weight_source_buffer(uid, t.get_element_type(), t.get_shape(), "CPU", handle);
        stored_tensor.tensor = tensor_factory(*m_core, "CPU", t.get_element_type(), t.get_shape(), shared_source);

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
        auto weight_shared_source_buffer = this->request_shared_weight_source_buffer(allocated.uid, allocated.meta.type, allocated.meta.shape, device, handle);

        ov::Tensor tensor = tensor_factory(*m_core, device, allocated.meta.type, allocated.meta.shape, weight_shared_source_buffer);
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
    NPUW_ASSERT(stream.output() && "Bank::serialize must use an output stream");
    LOG_INFO("Serializing weights bank...");
    LOG_BLOCK();

    if (this->serialize_native_format(stream)) {
        LOG_INFO("Bank has been serialized using OV weight sharing.");
        return;
    }
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

void Bank::deserialize(ov::npuw::orc::Stream& stream) {
    NPUW_ASSERT(stream.input() && "Bank::deserialize must use an input stream");
    LOG_INFO("Deserializing weights bank...");
    LOG_BLOCK();

    {
        auto handle = this->get_handle();
        if (this->deserialize_native_format_impl(stream, *m_core, handle)) {
            //return tensor_factory(core, device, type, shape, shared_source);
            std::unique_lock guard(m_mutex);
            for (const auto &[uid, weight_meta_map] : handle.context->m_weight_registry) {
                NPUW_ASSERT(weight_meta_map.size() == 1 && "Bank::deserialize: weight_meta_map must contain exactly one entry");
                auto weight_source = ov::weight_sharing::get_runtime_weight_source(*handle.context, uid);
                auto shared_source = weight_source.m_weights.lock();
                NPUW_ASSERT(shared_source && "Bank::deserialize: WeightSource must provide a valid shared source buffer");
                for (const auto& [constant_id, weight_meta] : weight_meta_map) {
                    NPUW_ASSERT(constant_id == 0 && "Bank::deserialize: constant id must be 0 for each weight transformation");
                    NPUW_ASSERT(weight_meta.m_offset == 0 && "Bank::deserialize: constant id must have m_offset 0 for each weight transformation");

                    // insert the restored data into the device bank for consistency
                    const std::string &device = weight_source.m_device;
                    auto constant_buffer = ov::weight_sharing::get_buffer(*handle.context, shared_source, constant_id);
                    const auto& ov_transformed_constant = this->get_transformed_constant(device, uid, handle);

                    auto& device_bank = m_device_banks[device];
                    auto iter_device = device_bank.storage.find(uid /* not constant_id! */);
                    NPUW_ASSERT(iter_device == device_bank.storage.end() && "Bank::deserialize: Tensor with this UID already exists in the device bank");
                    device_bank.storage[uid].tensor = tensor_factory(*m_core, weight_source.m_device, weight_meta.m_type, ov_transformed_constant.get_shape(), constant_buffer);
                    device_bank.storage[uid].lt = LazyTensor();

                }
            }
            LOG_INFO("Bank has been deserialized using OV weight sharing.");
            return;
        }
    }

    std::size_t bank_size;
    stream & bank_size;

    for (std::size_t i = 0; i < bank_size; ++i) {
        std::string device;
        std::size_t storage_size;
        stream & device & storage_size;

        for (std::size_t j = 0; j < storage_size; ++j) {
            int64_t uid;
            stream & uid;
            read_and_add_tensor(stream, uid, device);
        }
    }

    LOG_INFO("Deserialized in the legacy way.");
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

bool Bank::StoredTensor::is_evaluated() const {
    return static_cast<bool>(tensor);
}

void ov::npuw::orc::serialize(Stream& stream, ov::npuw::weights::Bank& var) {
    if (stream.output()) {
        var.serialize(stream);
    } else {
        var.deserialize(stream);
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
