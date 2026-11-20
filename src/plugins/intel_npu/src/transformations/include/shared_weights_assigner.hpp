// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
class Model;
namespace op {
namespace v0 {
class Constant;
}  // namespace v0
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_npu {
namespace transformations {

/**
 * @brief Collects model constants eligible for sharing and replaces them with constants backed by shared sources.
 *
 * Constants smaller than the system page size, larger than the configured per-source limit, or requiring
 * conversion for a configured device context are not shared.
 */
class SharedWeightsAssigner {
public:
    /// Model constant selected for sharing.
    using SharedConstant = std::shared_ptr<ov::op::v0::Constant>;
    /// Constants grouped into one shared source.
    using PartitionedConstants = std::vector<std::vector<SharedConstant>>;
    /// A shared source and the model constants backed by it.
    using SharedSourceAndConstants = std::pair<std::shared_ptr<ov::AlignedBuffer>, std::vector<SharedConstant>>;
    /// Shared sources paired with their constants.
    using SharedSourcesWithConstants = std::vector<SharedSourceAndConstants>;

    /// Configuration used when selecting and creating shared weight sources.
    struct Options {
        /// Device contexts whose data type requirements are considered when selecting constants.
        std::vector<std::string> shared_device_contexts;
        /// Maximum aligned size, in bytes, of one shared source buffer.
        size_t single_weight_shared_source_size_max = std::numeric_limits<size_t>::max();
        /// Whether to copy the weightless-cache runtime attribute to replacement constants.
        bool preserve_weightless_cache_attr = true;
        /// Generates source IDs for newly allocated shared source buffers.
        std::function<size_t()> source_id_generator;
    };

    /// Counts and byte totals describing collection and partitioning results.
    struct Statistic {
        /// Number of eligible constants in each shared source.
        std::vector<size_t> partition_constant_counts;
        /// Number of constants selected for sharing.
        size_t collected_constants_count = 0;
        /// Number of model constants rejected for sharing.
        size_t constant_cannot_be_shared_count = 0;
        /// Total aligned size, in bytes, of the constants in shared sources.
        size_t total_shared_constant_bytes = 0;
        /// Sum of original constant data sizes represented by shared constants.
        size_t total_non_shared_constant_bytes_released = 0;

        /// Sets the counts recorded during constant collection.
        /// @param collected_constants_count Number of constants selected for sharing.
        /// @param constant_cannot_be_shared_count Number of constants rejected for sharing.
        void set_collection_statistics(size_t collected_constants_count, size_t constant_cannot_be_shared_count);

        /// Computes per-partition counts and byte totals.
        /// @param partitioned_constants Constants grouped by shared source.
        /// @param alignment Alignment in bytes used for each constant in a source.
        void set_partition_statistics(const PartitionedConstants& partitioned_constants, size_t alignment);

        /// Formats the recorded statistics as a string.
        /// @return A human-readable representation of the statistics.
        std::string to_string() const;
    };

    /// Collection statistics and the constants grouped for sharing.
    struct CollectResult {
        Statistic statistic;
        PartitionedConstants partitioned_constants;
    };

    /// Creates an assigner with the supplied configuration.
    /// @param options Device compatibility, source sizing, and attribute-preservation options.
    explicit SharedWeightsAssigner(Options options);

    /// Selects constants eligible for sharing and groups them within the configured source-size limit.
    /// @param model Model whose constants are inspected.
    /// @return Collection statistics and the resulting constant partitions.
    CollectResult collect_and_partition(const std::shared_ptr<ov::Model>& model);

    /// Replaces each partition's constants in the model with constants backed by a shared source buffer.
    /// @param partitioned_constants Constants grouped by shared source, typically from collect_and_partition().
    /// @return Each allocated source buffer paired with the replacement constants backed by it.
    SharedSourcesWithConstants mutate_model_with_constant_sharing(PartitionedConstants&& partitioned_constants);

private:
    bool constant_can_be_shared(const ov::op::v0::Constant& constant) const;

    std::tuple<std::vector<SharedWeightsAssigner::SharedConstant>, size_t> collect_weights_to_share(
        const std::shared_ptr<ov::Model>& model) const;

    PartitionedConstants partition_constants_by_size(std::vector<SharedConstant>&& constants) const;

    std::shared_ptr<ov::AlignedBuffer> make_shared_source(const std::vector<SharedConstant>& partition) const;

    size_t get_constant_aligned_size(const ov::op::v0::Constant& constant) const;

    Options m_options;
    size_t m_min_relocate_bytes = 0;
};

}  // namespace transformations
}  // namespace intel_npu
}  // namespace ov
