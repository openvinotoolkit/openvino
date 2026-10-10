// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_weights_assigner.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>

#include "openvino/core/except.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/math_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace intel_npu {
namespace transformations {

void SharedWeightsAssigner::Statistic::set_collection_statistics(size_t collected_constants_count,
                                                                 size_t constant_cannot_be_shared_count) {
    this->collected_constants_count = collected_constants_count;
    this->constant_cannot_be_shared_count = constant_cannot_be_shared_count;
}

void SharedWeightsAssigner::Statistic::set_partition_statistics(const PartitionedConstants& partitioned_constants,
                                                                size_t alignment) {
    partition_constant_counts.clear();
    partition_constant_counts.reserve(partitioned_constants.size());
    total_shared_constant_bytes = 0;
    total_non_shared_constant_bytes_released = 0;

    for (const auto& partition : partitioned_constants) {
        partition_constant_counts.push_back(partition.size());
        for (const auto& constant : partition) {
            total_shared_constant_bytes += ov::util::align_size_up(constant->get_byte_size(), alignment);
            total_non_shared_constant_bytes_released += constant->get_byte_size();
        }
    }
}

std::string SharedWeightsAssigner::Statistic::to_string() const {
    std::ostringstream oss;
    oss << "collected_constants_count=" << collected_constants_count
        << ", constant_cannot_be_shared_count=" << constant_cannot_be_shared_count
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

SharedWeightsAssigner::SharedWeightsAssigner(Options options) : m_options(std::move(options)) {
    if (!m_options.source_id_generator) {
        m_options.source_id_generator = []() {
            // TODO The randomized counter start only reduces cross-process collision risk; it does not guarantee
            // unique IDs across processes or against mmap source IDs, and IDs are not stable across process runs.
            // Reliable cross-model weight sharing requires IDs that are persistent across processes, unique across
            // weight banks in one process, and distinguishable from mmap source IDs.
            static std::atomic<size_t> source_id_counter{[] {
                std::random_device random_device;
                std::uniform_int_distribution<size_t> distribution(1, std::numeric_limits<size_t>::max());
                return distribution(random_device);
            }()};
            return source_id_counter.fetch_add(1, std::memory_order_relaxed);
        };
    }
    m_min_relocate_bytes = static_cast<size_t>(::ov::util::get_system_page_size());
}

SharedWeightsAssigner::CollectResult SharedWeightsAssigner::collect_and_partition(
    const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model && "Model for assigning shared weights must not be null");
    OPENVINO_ASSERT(m_options.single_weight_shared_source_size_max > 0,
                    "single_weight_shared_source_size_max must be greater than zero");

    CollectResult result;

    auto [constants_to_share, constant_cannot_be_shared_count] = collect_weights_to_share(model);
    result.statistic.set_collection_statistics(constants_to_share.size(), constant_cannot_be_shared_count);

    result.partitioned_constants = partition_constants_by_size(std::move(constants_to_share));
    result.statistic.set_partition_statistics(result.partitioned_constants, m_min_relocate_bytes);

    return result;
}

SharedWeightsAssigner::SharedSourcesWithConstants SharedWeightsAssigner::mutate_model_with_constant_sharing(
    PartitionedConstants&& partitioned_constants) {
    SharedSourcesWithConstants shared_sources;
    for (const auto& partition : partitioned_constants) {
        auto shared_source = make_shared_source(partition);
        // By consideration with ov::Core, the constant ID is a weight offset in the shared source buffer
        // The offset allows distinguishing between different constants within the same shared source buffer.
        // Thus the offsets as the constant IDs serves two purposes:
        // 1. provides uniqueness of the constant in terms of source_id
        // 2. keeps consecutive constants properly ordered within the shared source buffer.
        size_t constant_id = 0;
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
            auto shared_constant = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(),
                                                                          constant->get_shape(),
                                                                          constant_shared_buffer);
            shared_constant->set_friendly_name(constant->get_friendly_name());
            ov::copy_runtime_info(constant, shared_constant);
            std::memcpy(constant_shared_buffer->get_ptr(), constant->get_data_ptr(), constant->get_byte_size());

            // Preserve the weightless-cache attribute: copy_runtime_info drops it (is_copyable()==false).
            if (m_options.preserve_weightless_cache_attr) {
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

bool SharedWeightsAssigner::constant_can_be_shared(const ov::op::v0::Constant& constant) const {
    if (constant.get_byte_size() < m_min_relocate_bytes ||
        constant.get_byte_size() > m_options.single_weight_shared_source_size_max) {
        return false;
    }

    bool needs_conversion = false;
    for (auto it = m_options.shared_device_contexts.begin();
         !needs_conversion && it != m_options.shared_device_contexts.end();
         ++it) {
        auto& device_context = *it;
        // Reconcile the constant's data type with the requirements of all shared device contexts
        if (device_context == "GPU") {
            if (constant.get_output_element_type(0) == ov::element::f64 ||
                constant.get_output_element_type(0) == ov::element::i64 ||
                constant.get_output_element_type(0) == ov::element::u32 ||
                constant.get_output_element_type(0) == ov::element::u16 ||
                constant.get_output_element_type(0) == ov::element::u4 ||
                constant.get_output_element_type(0) == ov::element::i64 ||
                constant.get_output_element_type(0) == ov::element::i16 ||
                constant.get_output_element_type(0) == ov::element::i4) {
                needs_conversion = true;
            }
        }
    }
    return !needs_conversion;
}

std::tuple<std::vector<SharedWeightsAssigner::SharedConstant>, size_t> SharedWeightsAssigner::collect_weights_to_share(
    const std::shared_ptr<ov::Model>& model) const {
    std::vector<SharedConstant> constants_to_share;
    size_t constant_cannot_be_shared_count = 0;
    for (const auto& op : model->get_ops()) {
        auto shared_weight_candidate = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
        if (!shared_weight_candidate) {
            continue;
        }

        if (!constant_can_be_shared(*shared_weight_candidate)) {
            ++constant_cannot_be_shared_count;
            continue;
        }
        constants_to_share.push_back(shared_weight_candidate);
    }
    return {constants_to_share, constant_cannot_be_shared_count};
}

SharedWeightsAssigner::PartitionedConstants SharedWeightsAssigner::partition_constants_by_size(
    std::vector<SharedConstant>&& constants) const {
    PartitionedConstants partitioned_constants;
    std::vector<SharedConstant> current_partition;
    size_t current_partition_size = 0;
    for (auto&& constant : constants) {
        size_t constant_size = get_constant_aligned_size(*constant);
        const size_t max_source_size = m_options.single_weight_shared_source_size_max;
        if (current_partition_size > max_source_size || constant_size > max_source_size - current_partition_size) {
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

std::shared_ptr<ov::AlignedBuffer> SharedWeightsAssigner::make_shared_source(
    const std::vector<SharedConstant>& partition) const {
    size_t total_partition_size = 0;
    for (const auto& constant : partition) {
        total_partition_size += get_constant_aligned_size(*constant);
    }

    const size_t source_id = m_options.source_id_generator();
    auto raw = std::make_shared<ov::AlignedBuffer>(total_partition_size, m_min_relocate_bytes);

    return std::make_shared<::ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        raw->get_ptr<char>(),
        raw->size(),
        raw,
        ::ov::create_base_descriptor(source_id, 0, raw));
}

size_t SharedWeightsAssigner::get_constant_aligned_size(const ov::op::v0::Constant& constant) const {
    return ov::util::align_size_up(constant.get_byte_size(), m_min_relocate_bytes);
}

}  // namespace transformations
}  // namespace intel_npu
}  // namespace ov
