// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weight_sharing_context_adapter.hpp"

#include <limits>
#include <map>
#include <numeric>
#include <thread>

#include "logging.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "orc.hpp"
#include "serialization.hpp"
#include "util.hpp"

using ov::npuw::weights::TransformationWeightContextIDMapper;
using ov::npuw::weights::WeightSharingContextAccessor;

namespace {
class ContextIDsGuard {
public:
    explicit ContextIDsGuard(std::atomic_flag& flag) : m_flag(flag) {
        while (m_flag.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    ~ContextIDsGuard() {
        m_flag.clear(std::memory_order_release);
    }

    ContextIDsGuard(const ContextIDsGuard&) = delete;
    ContextIDsGuard& operator=(const ContextIDsGuard&) = delete;

private:
    std::atomic_flag& m_flag;
};

/* TODO CVS-199040:
 * Proper implementation of ov::IBufferDescriptor for ov::Tensor.
 * Once that implemented, the entire struct can be removed
 */
struct SharedTensorDescriptor : public ov::IBufferDescriptor {
    size_t get_id() const override {
        return m_source_id;
    }

    size_t get_offset() const override {
        return m_offset;
    }

    std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const override {
        // as long as we store tensor by ourselves,
        // we can safely create a shared buffer from it every time someone asks for the source buffer.
        //
        // I don't want to store the tensor in WeightSharingContextAccessor, because
        // it is not the responsibility of WeightSharingContextAccessor to manage the lifetime of the tensor.
        // The CVS-199040 should address the proper handling of tensor lifetimes in this context,
        // until then we store the tensor here in a temporary place.
        return std::make_shared<ov::SharedBuffer<ov::Tensor>>(
            const_cast<char*>(static_cast<const char*>(m_tensor.data())),
            m_tensor.get_byte_size(),
            m_tensor);
    }

    const ov::Tensor& get_tensor() const {
        return m_tensor;
    }

    ~SharedTensorDescriptor() = default;

    static std::shared_ptr<SharedTensorDescriptor> create(size_t source_id, size_t offset, const ov::Tensor& tensor) {
        return std::shared_ptr<SharedTensorDescriptor>(new SharedTensorDescriptor(source_id, offset, tensor));
    }

private:
    SharedTensorDescriptor(size_t source_id, size_t offset, const ov::Tensor& tensor)
        : m_source_id(source_id),
          m_offset(offset),
          m_tensor(tensor) {}
    size_t m_source_id;
    size_t m_offset;
    ov::Tensor m_tensor;
};

// TODO CVS-199040: Preserve the tensor descriptor until ov::Tensor supports IBufferDescriptor.
// or
// CVS-199039: In this case we woudln't need for SharedTensorDescriptor at all,
// the HostTensor could be created directly from the staging buffer obtained by its source_id
class SharedTensorBuffer final : public ov::SharedBuffer<ov::Tensor> {
public:
    using ov::SharedBuffer<ov::Tensor>::SharedBuffer;

    std::shared_ptr<ov::IBufferDescriptor> get_descriptor() const override {
        if (!m_descriptor) {
            return {};
        }
        return SharedTensorDescriptor::create(m_descriptor->get_id(), m_descriptor->get_offset(), m_shared_object);
    }
};

}  // namespace

ov::Tensor ov::npuw::weights::tensor_factory(const ov::ICore& core,
                                             const std::string& device,
                                             const ov::element::Type& type,
                                             const ov::Shape& shape,
                                             std::shared_ptr<ov::AlignedBuffer> staging_buffer_ptr) {
    if (device == "CPU") {
        return staging_buffer_ptr ? ov::Tensor(type, shape, staging_buffer_ptr->get_ptr<char>())
                                  : ov::Tensor(type, shape);
    }

    auto remote_ctx = core.get_default_context(device)._ptr;
    if (ov::npuw::util::starts_with(device, "NPU")) {
        // TODO CVS-199039, CVS-199076
        // Cannot create neither a host tensor from preexisting staging buffer nor a remote tensor for NPU remote
        // context
        NPUW_ASSERT(!staging_buffer_ptr && "create_tensor_allocator: NPU remote context cannot use the existing "
                                           "staging buffer for its host tensor creation");
    } else if (ov::npuw::util::starts_with(device, "GPU")) {
        // TODO CVS-199039
        NPUW_ASSERT(!staging_buffer_ptr && "create_tensor_allocator: GPU remote context cannot use  the existing "
                                           "staging buffer for its host tensor creation");
    }

    return ov::make_tensor(remote_ctx->create_host_tensor(type, shape));
}

ov::Tensor ov::npuw::weights::WeightSharingContextAccessor::tensor_from_shared_buffer(
    const ov::weight_sharing::WeightBuffer& shared_source) {
    auto source_descriptor = shared_source->get_descriptor();
    auto shared_descriptor = std::dynamic_pointer_cast<SharedTensorDescriptor>(source_descriptor);
    NPUW_ASSERT(shared_descriptor && "WeightSharingContextAccessor: Failed to cast to SharedTensorDescriptor");
    return shared_descriptor->get_tensor();
}

std::optional<TransformationWeightContextIDMapper::WSContextIDs>
ov::npuw::weights::TransformationWeightContextIDMapper::get_ov_context_ids(UID_Type uid) const {
    const ContextIDsGuard guard(m_context_ids_lock);
    const auto context_ids = m_ov_context_ids.find(uid);
    if (context_ids == m_ov_context_ids.end()) {
        return std::nullopt;
    }
    return context_ids->second;
}

std::tuple<bool, TransformationWeightContextIDMapper::WSContextIDs>
ov::npuw::weights::TransformationWeightContextIDMapper::get_or_create_ov_context_ids(UID_Type uid) {
    const ContextIDsGuard guard(m_context_ids_lock);
    const auto existing = m_ov_context_ids.find(uid);
    if (existing != m_ov_context_ids.end()) {
        return {true, existing->second};
    }
    const auto context_ids = create_ov_context_ids(uid);
    insert_mapping(uid, context_ids);
    return {false, context_ids};
}

TransformationWeightContextIDMapper::UID_Type
ov::npuw::weights::TransformationWeightContextIDMapper::get_transformation_uid(const WSContextIDs& context_ids) const {
    const ContextIDsGuard guard(m_context_ids_lock);
    return m_transformation_uids.at(context_ids);
}

TransformationWeightContextIDMapper::WSContextIDs
ov::npuw::weights::TransformationWeightContextIDMapper::create_ov_context_ids(UID_Type uid) {
    OPENVINO_ASSERT(uid >= 0, "Transformation UID must be nonnegative: ", uid);
    return {static_cast<SourceID_Type>(uid) + 1, 0};
}

TransformationWeightContextIDMapper::WSContextIDs
ov::npuw::weights::TransformationWeightContextIDMapper::set_ov_context_ids(UID_Type uid) {
    const auto context_ids = create_ov_context_ids(uid);
    set_transformation_uid(context_ids);
    return context_ids;
}

TransformationWeightContextIDMapper::UID_Type
ov::npuw::weights::TransformationWeightContextIDMapper::set_transformation_uid(const WSContextIDs& context_ids) {
    const auto source_id = context_ids.first;
    OPENVINO_ASSERT(source_id > 0 && source_id - 1 <= static_cast<SourceID_Type>(std::numeric_limits<UID_Type>::max()),
                    "Source ID cannot be mapped to a transformation UID: ",
                    source_id);
    const auto uid = static_cast<UID_Type>(source_id - 1);
    const ContextIDsGuard guard(m_context_ids_lock);
    insert_mapping(uid, context_ids);
    return uid;
}

void ov::npuw::weights::TransformationWeightContextIDMapper::insert_mapping(UID_Type uid,
                                                                            const WSContextIDs& context_ids) {
    const auto [forward_it, inserted] = m_ov_context_ids.emplace(uid, context_ids);
    OPENVINO_ASSERT(forward_it->second == context_ids,
                    "Transformation UID already maps to different context IDs: ",
                    uid);
    try {
        m_transformation_uids.emplace(context_ids, uid);
    } catch (...) {
        if (inserted) {
            m_ov_context_ids.erase(forward_it);
        }
        throw;
    }
}

ov::weight_sharing::WeightBuffer ov::npuw::weights::WeightSharingContextAccessor::create_shared_buffer(
    ov::weight_sharing::DataID source_id,
    std::size_t size) {
    // Create page aligned buffer to satisfy further GPU-NPU memory sharing requirements
    auto raw = std::make_shared<ov::AlignedBuffer>(size, ov::util::get_system_page_size());
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        raw->get_ptr<char>(),
        raw->size(),
        raw,
        ov::create_base_descriptor(source_id, 0, raw));
}

ov::weight_sharing::WeightBuffer ov::npuw::weights::WeightSharingContextAccessor::create_shared_buffer(
    ov::weight_sharing::DataID source_id,
    const ov::Tensor& tensor) {
    return std::make_shared<SharedTensorBuffer>(const_cast<char*>(static_cast<const char*>(tensor.data())),
                                                tensor.get_byte_size(),
                                                tensor,
                                                SharedTensorDescriptor::create(source_id, 0, tensor));
}

const ov::op::v0::Constant& ov::npuw::weights::WeightSharingContextAccessor::get_transformed_constant(
    const std::string& device,
    UID_Type transformation_uid,
    Handle& handle) const {
    OPENVINO_ASSERT(handle.context, "WeightSharingContextAccessor has no weight sharing context");
    auto dev_it = m_transformed_weights_per_device.find(device);
    OPENVINO_ASSERT(dev_it != m_transformed_weights_per_device.end(),
                    "No transformed weights found for device: ",
                    device);
    auto iter = dev_it->second.find(transformation_uid);
    OPENVINO_ASSERT(iter != dev_it->second.end(),
                    "Transformed constant not found for transformation UID: ",
                    transformation_uid,
                    ", on device: ",
                    device);
    return iter->second;
}

bool ov::npuw::weights::WeightSharingContextAccessor::has_transformation_buffer(UID_Type transformation_uid,
                                                                                Handle& handle) {
    if (!handle.context) {
        return false;
    }

    const auto context_ids = ids_mapper.get_ov_context_ids(transformation_uid);
    if (!context_ids) {
        return false;
    }
    const auto [source_id, constant_id] = *context_ids;
    NPUW_ASSERT(constant_id == 0 &&
                "has_transformation_buffer: Only constant_id 0 is supported for shared weight sources");
    auto shared_weight_source = ov::weight_sharing::get_runtime_weight_source(*handle.context, source_id);
    return static_cast<bool>(shared_weight_source.m_weights.lock());
}

ov::weight_sharing::WeightBuffer ov::npuw::weights::WeightSharingContextAccessor::request_shared_weight_source_buffer(
    UID_Type transformation_uid,
    const element::Type& type,
    const ov::Shape& shape,
    const std::string& device,
    Handle& handle) {
    if (!handle.context) {
        return {};
    }

    auto size = ov::util::get_memory_size_safe(type, shape);
    OPENVINO_ASSERT(size.has_value(), "LazyTensor size overflow, uid: ", transformation_uid, " device: ", device);

    auto [existing, context_ids] = ids_mapper.get_or_create_ov_context_ids(transformation_uid);
    const auto [source_id, constant_id] = context_ids;
    NPUW_ASSERT(constant_id == 0 &&
                "request_shared_weight_source_buffer: Only constant_id 0 is supported for shared weight sources");
    LOG_DEBUG("WeightSharingContextAccessor::request_shared_weight_source_buffer - source id: "
              << source_id << ", device: " << device << ", size: " << *size << ", existed before: " << existing);

    auto shared_weight_source = ov::weight_sharing::get_runtime_weight_source(*handle.context, source_id);
    if (auto shared_buffer = shared_weight_source.m_weights.lock(); shared_buffer != nullptr) {
        OPENVINO_ASSERT(shared_weight_source.m_device == device,
                        "Shared weight source exists for the wrong device, expected: ",
                        device,
                        " actual: ",
                        shared_weight_source.m_device);
        LOG_VERB(
            "WeightSharingContextAccessor::request_shared_weight_source_buffer - reusing shared buffer for source id: "
            << source_id << ", device: " << device);
        return shared_buffer;
    }

    auto shared_buffer = WeightSharingContextAccessor::create_shared_buffer(source_id, *size);
    register_shared_weight_source_for_weight(shared_buffer, device, handle);
    register_transformed_constant(shared_buffer, transformation_uid, device, type, shape, handle);
    return shared_buffer;
}

ov::weight_sharing::WeightBuffer
ov::npuw::weights::WeightSharingContextAccessor::register_tensor_as_shared_weight_source(UID_Type transformation_uid,
                                                                                         const std::string& device,
                                                                                         const ov::Tensor& tensor,
                                                                                         Handle& handle) {
    if (!handle.context) {
        return {};
    }

    auto [existing, context_ids] = ids_mapper.get_or_create_ov_context_ids(transformation_uid);
    const auto [source_id, constant_id] = context_ids;
    NPUW_ASSERT(constant_id == 0 &&
                "register_tensor_as_shared_weight_source: Only constant_id 0 is supported for shared weight sources");

    LOG_DEBUG("WeightSharingContextAccessor::register_tensor_as_shared_weight_source - source id: "
              << source_id << ", device: " << device << ", size: " << tensor.get_size()
              << ", existed before: " << existing);

    auto shared_buffer = WeightSharingContextAccessor::create_shared_buffer(source_id, tensor);
    register_shared_weight_source_for_weight(shared_buffer, device, handle);
    register_transformed_constant(shared_buffer,
                                  transformation_uid,
                                  device,
                                  tensor.get_element_type(),
                                  tensor.get_shape(),
                                  handle);
    return shared_buffer;
}

void ov::npuw::weights::WeightSharingContextAccessor::register_transformed_constant(
    std::shared_ptr<ov::AlignedBuffer> transformation_buffer,
    UID_Type transformation_uid,
    const std::string& device,
    const element::Type& type,
    const ov::Shape& shape,
    Handle& handle) {
    NPUW_ASSERT(transformation_buffer &&
                "WeightSharingContextAccessor::register_transformed_constant - Transformation buffer must not be null");

    // A transformed constant is never exist in the model as a node.
    // let's create a fake constant using the shared buffer as its data source
    // and register it with the weight sharing context.
    // The current mapping of `uid` to `source_id` and `0` as `constant id` is intendional
    // and must be treated as the private to NPUW.
    //
    // The generic mapping imposed by OV Core as `source_id` to a unique staging buffer and
    // as `constant_id` to an offset within that staging buffer is not applicable here without
    // huge modification. TO satisfy those mapping requirements, NPUW may introduce
    // a new phase in addition to registerLT() and evaluate() as estimate(),
    // which should pre-evaluate LazyTensors by counting a total size of the contiguous memory
    // required to accomodate all evaluated tensors for the closure. After that we
    // can allocate the aligned staging buffer of this total size, assign `source_id` to it,
    // and use `constant_id` to map to the offset within that staging buffer.
    auto transformed_constant = ov::op::v0::Constant(type, shape, transformation_buffer);
    ov::weight_sharing::set_constant(*handle.context, transformed_constant);

    // CVS-198039 WeightMetaData has no m_shape
    // As weight sharing context doesn't include the shape of the constant in WeightMetaData,
    // we store and remember the transformed constant,
    // so that we can deserialize its ov::Tensor from raw buffer.
    m_transformed_weights_per_device[device].erase(transformation_uid);
    m_transformed_weights_per_device[device].emplace(transformation_uid, std::move(transformed_constant));
    LOG_DEBUG("WeightSharingContextAccessor::register_transformed_constant - m_transformed_weights_per_device: "
              << device << ", counts: " << m_transformed_weights_per_device[device].size());
}

bool ov::npuw::weights::WeightSharingContextAccessor::serialize_native_format(ov::npuw::orc::Stream& stream) {
    auto lock = get_handle();
    return serialize_native_format_impl(stream, lock);
}

bool ov::npuw::weights::WeightSharingContextAccessor::serialize_native_format_impl(ov::npuw::orc::Stream& stream,
                                                                                   Handle& handle) const {
    using namespace ov::weight_sharing;
    NPUW_ASSERT(stream.output() &&
                "WeightSharingContextAccessor::serialize_native_format_impl must receive an output stream");
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
    LOG_DEBUG("WeightSharingContextAccessor::serialize_native_format_impl - total devices: " << bank_size);
    stream & bank_size;
    for (const auto& [device, source_ids] : weight_sources_per_device) {
        // 1) serialize the name of the device
        stream & device;

        // 2) serialize the number of UIDs for this device
        size_t num_weights = 0;
        num_weights = std::accumulate(constant_ids_per_source_id_ordered.begin(),
                                      constant_ids_per_source_id_ordered.end(),
                                      num_weights,
                                      [&device_source_ids = source_ids](size_t sum, const auto& pair) {
                                          return device_source_ids.count(pair.first) ? sum + pair.second.size() : sum;
                                      });
        LOG_DEBUG("WeightSharingContextAccessor::serialize_native_format_impl - total weights for device "
                  << device << ": " << num_weights);
        stream & num_weights;

        // serialize weights in the incrementing order
        for (const auto& [source_id, constant_ids] : constant_ids_per_source_id_ordered) {
            if (!source_ids.count(source_id))
                continue;

            // for NPUW we must not serialize constant_ids, which number is always 1,
            // and the only element has constant_id equals to 0 for each weight transformation
            NPUW_ASSERT(constant_ids.size() == 1 &&
                        "NPUW: transformed constant ids for each source must have exactly one constant id");
            for (auto constant_id : constant_ids) {
                NPUW_ASSERT(constant_id == 0 && "NPUW: constant id must be 0 for each weight transformation");
                auto uid = ids_mapper.get_transformation_uid({source_id, constant_id});
                stream & uid;
                const WeightMetaData& weight_meta_map =
                    m_context->m_weight_registry.find(source_id)->second.find(constant_id)->second;
                bool is_initialized =
                    true;  // once the weight is found in m_weight_registry, it is considered initialized
                stream & is_initialized;

                // 4) serialize the weight metadata
                auto type_str = weight_meta_map.m_type.to_string();
                /*CVS-198039 WeightMetaData has no m_shape*/
                const auto& transformed_constant = get_transformed_constant(device, uid, handle);
                auto shape = transformed_constant.get_shape();
                auto byte_size = weight_meta_map.m_size;
                stream & type_str & shape & byte_size;

                // 5) finally serialize the actual weight data
                const WeightSource& data_source = m_context->m_runtime_sources.find(source_id)->second;
                auto source_staging_buffer = data_source.m_weights.lock();
                OPENVINO_ASSERT(source_staging_buffer && "WeightSharingContextAccessor: staging buffer for source id: ",
                                source_id,
                                " must exist");
                auto source_staging_buffer_descriptor = source_staging_buffer->get_descriptor();
                OPENVINO_ASSERT(
                    source_staging_buffer_descriptor && "WeightSharingContextAccessor: staging buffer for source id: ",
                    source_id,
                    " must have a valid descriptor");
                LOG_VERB(
                    "WeightSharingContextAccessor::serialize_native_format_impl - serializing weight data for uid: "
                    << uid << ", source_id: " << source_staging_buffer_descriptor->get_id()
                    << ", size: " << source_staging_buffer->size());
                stream.bytes(source_staging_buffer->get_ptr(weight_meta_map.m_offset), weight_meta_map.m_size);
            }
        }
    }
    return true;
}

std::set<WeightSharingContextAccessor::UID_Type> ov::npuw::weights::WeightSharingContextAccessor::consume_from(
    ov::npuw::orc::Stream& stream,
    const ov::ICore& core) {
    auto lock = get_handle();
    return consume_from_impl(stream, core, lock);
}

std::set<WeightSharingContextAccessor::UID_Type> ov::npuw::weights::WeightSharingContextAccessor::consume_from_impl(
    ov::npuw::orc::Stream& stream,
    const ov::ICore& core,
    Handle& handle) {
    NPUW_ASSERT(stream.input() && "WeightSharingContextAccessor::consume_from_impl must receive an input stream");

    // TODO who will create the m_context if it is not already created?
    if (!handle.context) {
        return {};
    }

    std::set<UID_Type> deserialized_uids;
    size_t bank_size = 0;
    stream & bank_size;
    LOG_DEBUG("WeightSharingContextAccessor::consume_from_impl - total devices: " << bank_size);
    for (size_t i = 0; i < bank_size; ++i) {
        std::string device;
        stream & device;
        std::size_t storage_size = 0;
        stream & storage_size;
        LOG_DEBUG("WeightSharingContextAccessor::consume_from_impl - total weights for device " << device << ": "
                                                                                                << storage_size);
        for (std::size_t j = 0; j < storage_size; ++j) {
            UID_Type uid = -1;
            stream & uid;

            auto common_allocator = [this, &core, &device, uid, &handle](const ov::element::Type& type,
                                                                         const ov::Shape& shape) {
                NPUW_ASSERT(
                    !this->has_transformation_buffer(uid, handle) &&
                    "WeightSharingContextAccessor::consume_from_impl: Shared weight source must not already exist");
                if (device == "CPU") {
                    auto shared_source = this->request_shared_weight_source_buffer(uid, type, shape, device, handle);
                    return tensor_factory(core, device, type, shape, shared_source);
                }

                ov::Tensor tensor = tensor_factory(core, device, type, shape, nullptr);
                this->register_tensor_as_shared_weight_source(uid, device, tensor, handle);
                return tensor;
            };

            ov::Tensor regained_tensor;
            transfer_tensor(stream, regained_tensor, common_allocator);
            (void)
                regained_tensor;  // can be dropped, not needed. It can be restored from source_descriptor if necessary.
            deserialized_uids.insert(uid);
            LOG_DEBUG("WeightSharingContextAccessor::consume_from_impl - weight data restored, uid: " << uid);
        }
    }
    return deserialized_uids;
}

void ov::npuw::weights::WeightSharingContextAccessor::register_shared_weight_source_for_weight(
    ov::weight_sharing::WeightBuffer source_buffer,
    const std::string& device,
    Handle& handle) {
    if (!handle.context) {
        return;
    }

    auto buf_descriptor = source_buffer->get_descriptor();
    OPENVINO_ASSERT(buf_descriptor != nullptr,
                    "WeightSharingContextAccessor::register_shared_weight_source_for_weight - must register a source "
                    "buffer, that have a valid descriptor");
    ov::weight_sharing::DataID source_id = buf_descriptor->get_id();
    m_shared_sources[source_id] = source_buffer;
    ov::weight_sharing::set_runtime_weight_source(*handle.context, source_buffer, device);
    LOG_DEBUG("WeightSharingContextAccessor::register_shared_weight_source_for_weight - source id: "
              << source_id << ", device: " << device);
}
