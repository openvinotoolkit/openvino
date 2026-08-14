// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_weights_assigner.hpp"

#include <atomic>
#include <cstring>
#include <sstream>

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/openvino.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace intel_npu {

std::string SharedWeightsAssigner::Statistic::to_string() const {
    std::ostringstream oss;
    oss << "collected_constants_count=" << collected_constants_count
        << ", total_shared_constant_bytes=" << total_shared_constant_bytes
        << ", total_non_shared_constant_bytes_released=" << total_non_shared_constant_bytes_released
        << ", partitions=" << partition_constant_counts.size() << " [";
    for (size_t i = 0; i < partition_constant_counts.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << partition_constant_counts[i];
    }
    oss << "]";
    return oss.str();
}

SharedWeightsAssigner::SharedWeightsAssigner(Options options)
        : m_shared_device_contexts(std::move(options.shared_device_contexts)),
            m_source_id_generator(std::move(options.source_id_generator)),
            m_preserve_weightless_cache_attr(options.preserve_weightless_cache_attr),
            m_single_weight_shared_source_size_max(options.single_weight_shared_source_size_max) {
        if (!m_source_id_generator) {
                m_source_id_generator = []() {
                        static std::atomic<size_t> source_id_counter{1};
                        return source_id_counter.fetch_add(1, std::memory_order_relaxed);
                };
        }
    m_min_relocate_bytes = static_cast<size_t>(::ov::util::get_system_page_size());
}

SharedWeightsAssigner::CollectResult SharedWeightsAssigner::collect_and_partition(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model && "Model for assigning shared weights must not be null");
    OPENVINO_ASSERT(m_single_weight_shared_source_size_max > 0,
                    "single_weight_shared_source_size_max must be greater than zero");

    CollectResult result;

    auto constants_to_share = collect_weights_to_share(model);
    result.statistic.collected_constants_count = constants_to_share.size();

    result.partitioned_constants = partition_constants_by_size(std::move(constants_to_share));
    result.statistic.partition_constant_counts.reserve(result.partitioned_constants.size());
    for (const auto& partition : result.partitioned_constants) {
        result.statistic.partition_constant_counts.push_back(partition.size());
        for (const auto& constant : partition) {
            result.statistic.total_shared_constant_bytes += get_constant_aligned_size(*constant);
            result.statistic.total_non_shared_constant_bytes_released += constant->get_byte_size();
        }
    }

    return result;
}

SharedWeightsAssigner::SharedSourcesWithConstants SharedWeightsAssigner::mutate_model_with_constant_sharing(
    PartitionedConstants&& partitioned_constants) {
    SharedSourcesWithConstants shared_sources_with_constants;
    shared_sources_with_constants.clear();
    shared_sources_with_constants = make_constant_shareable(std::move(partitioned_constants));
    return shared_sources_with_constants;
}

bool SharedWeightsAssigner::constant_can_be_shared(const ov::op::v0::Constant& constant) const {
    if (constant.get_byte_size() < m_min_relocate_bytes ||
        constant.get_byte_size() > m_single_weight_shared_source_size_max) {
        return false;
    }

    bool needs_conversion = false;
    // TODO check types of remote context
    // The code below is written in assumption that we have NPU and GPU as the remote contexts
    (void)m_shared_device_contexts;
    if (ov::shape_size(constant.get_shape()) == 1 && constant.get_output_element_type(0) == ov::element::f64) {
        // If a constant has element type f64 but contains no elements (empty tensor),
        // GPU have to convert it to f32 because the GPU plugin only supports the f32 data type internally.
        needs_conversion = true;
    } else if (constant.get_output_element_type(0) == ov::element::u16 ||
               constant.get_output_element_type(0) == ov::element::i16) {
        needs_conversion = true;
    }
    return !needs_conversion;
}

std::vector<SharedWeightsAssigner::SharedConstant> SharedWeightsAssigner::collect_weights_to_share(
    const std::shared_ptr<ov::Model>& model) const {
    std::vector<SharedConstant> constants_to_share;
    for (const auto& op : model->get_ops()) {
        auto shared_weight_candidate = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
        if (!shared_weight_candidate) {
            continue;
        }

        if (!constant_can_be_shared(*shared_weight_candidate)) {
            continue;
        }
        constants_to_share.push_back(shared_weight_candidate);
    }
    return constants_to_share;
}

SharedWeightsAssigner::PartitionedConstants SharedWeightsAssigner::partition_constants_by_size(
    std::vector<SharedConstant>&& constants) const {
    PartitionedConstants partitioned_constants;
    std::vector<SharedConstant> current_partition;
    size_t current_partition_size = 0;
    for (auto&& constant : constants) {
        size_t constant_size = get_constant_aligned_size(*constant);
        if (current_partition_size + constant_size > m_single_weight_shared_source_size_max) {
            if (!current_partition.empty()) {
                partitioned_constants.emplace_back(std::move(current_partition));
                current_partition.clear();
                current_partition_size = 0;
            }
        }
        current_partition.push_back(constant);
        current_partition_size += constant_size;
    }
    if (!current_partition.empty()) {
        partitioned_constants.emplace_back(std::move(current_partition));
    }
    return partitioned_constants;
}

SharedWeightsAssigner::SharedSourcesWithConstants SharedWeightsAssigner::make_constant_shareable(
    PartitionedConstants&& partitioned_constants) const {
    SharedSourcesWithConstants shared_sources;
    for (const auto& partition : partitioned_constants) {
        auto shared_source = make_shared_source(partition);
        size_t constant_id = 0;  // constants ID is a weight offset in the shared source buffer
        std::vector<SharedConstant> shared_constants;
        for (const auto& constant : partition) {
            auto const_descriptor =
                ::ov::create_base_descriptor(shared_source->get_descriptor()->get_id(), constant_id, shared_source);
            auto constant_shared_buffer = std::make_shared<::ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
                shared_source->get_ptr<char>() + constant_id,
                constant->get_byte_size(),
                shared_source,
                const_descriptor);
            constant_id += get_constant_aligned_size(*constant);
            auto shared_constant =
                std::make_shared<ov::op::v0::Constant>(constant->get_element_type(), constant->get_shape(), constant_shared_buffer);
            shared_constant->set_friendly_name(constant->get_friendly_name());
            ov::copy_runtime_info(constant, shared_constant);
            std::memcpy(constant_shared_buffer->get_ptr(), constant->get_data_ptr(), constant->get_byte_size());

            // Preserve the weightless-cache attribute: copy_runtime_info drops it (is_copyable()==false).
            if (m_preserve_weightless_cache_attr) {
                ov::copy_weightless_cache_attr(constant, shared_constant);
            }

            ov::replace_node(constant, shared_constant);
            ov::weight_sharing::Extension::hint_evict(*constant);
            shared_constants.push_back(shared_constant);
        }
        shared_sources.emplace_back(std::move(shared_source), std::move(shared_constants));
    }
    return shared_sources;
}

std::shared_ptr<ov::AlignedBuffer> SharedWeightsAssigner::make_shared_source(
    const std::vector<SharedConstant>& partition) const {
    size_t total_partition_size = 0;
    for (const auto& constant : partition) {
        total_partition_size += get_constant_aligned_size(*constant);
    }

    // TODO not a unique ID in general: as it can clash with mmap weight source if generation.
    // The uniqueness must meet the conditions:
    // 1) persistent across different processes.
    // 2) unique across different weight banks in the same process.
    // 3) distinguishable from mmap sources for weightless cache.
    const size_t source_id = m_source_id_generator();
    auto raw = std::make_shared<ov::AlignedBuffer>(total_partition_size, m_min_relocate_bytes);

    return std::make_shared<::ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        raw->get_ptr<char>(),
        raw->size(),
        raw,
        ::ov::create_base_descriptor(source_id, 0, raw));
}

size_t SharedWeightsAssigner::get_constant_aligned_size(const ov::op::v0::Constant& constant) const {
    return align_bytes(constant.get_byte_size(), m_min_relocate_bytes);
}

size_t SharedWeightsAssigner::align_bytes(size_t bytes, size_t alignment) {
    return ((bytes + alignment - 1) / alignment) * alignment;
}

}  // namespace intel_npu
}  // namespace ov
