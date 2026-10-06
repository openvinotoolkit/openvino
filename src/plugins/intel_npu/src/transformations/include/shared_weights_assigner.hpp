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

class SharedWeightsAssigner {
public:
    using SharedConstant = std::shared_ptr<ov::op::v0::Constant>;
    using PartitionedConstants = std::vector<std::vector<SharedConstant>>;
    using SharedSourceAndConstants = std::pair<std::shared_ptr<ov::AlignedBuffer>, std::vector<SharedConstant>>;
    using SharedSourcesWithConstants = std::vector<SharedSourceAndConstants>;

    struct Options {
        std::vector<std::string> shared_device_contexts;
        size_t single_weight_shared_source_size_max = std::numeric_limits<size_t>::max();
        bool preserve_weightless_cache_attr = true;
        std::function<size_t()> source_id_generator;
    };

    struct Statistic {
        std::vector<size_t> partition_constant_counts;
        size_t collected_constants_count = 0;
        size_t constant_cannot_be_shared_count = 0;
        size_t total_shared_constant_bytes = 0;
        size_t total_non_shared_constant_bytes_released = 0;

        void set_collection_statistics(size_t collected_constants_count, size_t constant_cannot_be_shared_count);
        void set_partition_statistics(const PartitionedConstants& partitioned_constants, size_t alignment);
        std::string to_string() const;
    };

    struct CollectResult {
        Statistic statistic;
        PartitionedConstants partitioned_constants;
    };

    explicit SharedWeightsAssigner(Options options);
    CollectResult collect_and_partition(const std::shared_ptr<ov::Model>& model);
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
