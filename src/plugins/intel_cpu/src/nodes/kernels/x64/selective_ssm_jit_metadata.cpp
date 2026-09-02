// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_jit_metadata.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "nodes/kernels/x64/selective_ssm_jit_runtime.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {
namespace {

template <typename Index>
size_t checked_block_index(Index value, size_t block_count, size_t position) {
    OPENVINO_ASSERT(value >= 0,
                    "PagedSelectiveSSM: block_indices[",
                    position,
                    "] must be non-negative, got ",
                    value,
                    ".");
    const auto index = static_cast<uint64_t>(value);
    OPENVINO_ASSERT(index < block_count,
                    "PagedSelectiveSSM: block_indices[",
                    position,
                    "] is out of range: ",
                    value,
                    " not in [0, ",
                    block_count,
                    ").");
    return static_cast<size_t>(index);
}

template <typename Index>
void validate(const PagedSelectiveSSMJitRuntimeArgs& args) {
    const auto& shape = args.shape;
    const auto* subsequence_begins = static_cast<const Index*>(args.subsequence_begins);
    const auto* block_indices = static_cast<const Index*>(args.block_indices);
    const auto* block_indices_begins = static_cast<const Index*>(args.block_indices_begins);
    const auto* num_processed_tokens = static_cast<const Index*>(args.num_processed_tokens);
    const auto* cache_intervals = static_cast<const Index*>(args.cache_intervals);
    auto* block_owners = args.metadata_validation_scratch;

    OPENVINO_ASSERT(shape.sequence_count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                    "PagedSelectiveSSM supports at most INT32_MAX sequences.");
    OPENVINO_ASSERT(shape.physical_block_count == 0 || block_owners != nullptr,
                    "PagedSelectiveSSM JIT requires block-owner scratch for a non-empty state table.");
    OPENVINO_ASSERT(subsequence_begins[0] == 0,
                    "PagedSelectiveSSM: subsequence_begins[0] must be 0, got ",
                    subsequence_begins[0],
                    ".");
    OPENVINO_ASSERT(block_indices_begins[0] == 0,
                    "PagedSelectiveSSM: block_indices_begins[0] must be 0, got ",
                    block_indices_begins[0],
                    ".");

    const auto final_token_offset = subsequence_begins[shape.sequence_count];
    OPENVINO_ASSERT(final_token_offset >= 0 && static_cast<uint64_t>(final_token_offset) == shape.token_count,
                    "PagedSelectiveSSM: the last subsequence offset must equal token_count (",
                    shape.token_count,
                    "), got ",
                    final_token_offset,
                    ".");
    const auto final_block_offset = block_indices_begins[shape.sequence_count];
    OPENVINO_ASSERT(final_block_offset >= 0 && static_cast<uint64_t>(final_block_offset) == shape.logical_block_count,
                    "PagedSelectiveSSM: the last block offset must equal logical_block_count (",
                    shape.logical_block_count,
                    "), got ",
                    final_block_offset,
                    ".");

    if (shape.physical_block_count > 0) {
        std::fill(block_owners, block_owners + shape.physical_block_count, int32_t{-1});
    }

    for (size_t sequence = 0; sequence < shape.sequence_count; ++sequence) {
        const auto token_begin = subsequence_begins[sequence];
        const auto token_end = subsequence_begins[sequence + 1];
        const auto block_begin_value = block_indices_begins[sequence];
        const auto block_end_value = block_indices_begins[sequence + 1];
        const auto processed_tokens = num_processed_tokens[sequence];
        OPENVINO_ASSERT(token_begin >= 0 && token_end >= token_begin,
                        "PagedSelectiveSSM: subsequence_begins must be non-negative and non-decreasing at sequence ",
                        sequence,
                        ".");
        OPENVINO_ASSERT(block_begin_value >= 0 && block_end_value >= block_begin_value,
                        "PagedSelectiveSSM: block_indices_begins must be non-negative and non-decreasing at sequence ",
                        sequence,
                        ".");
        OPENVINO_ASSERT(processed_tokens >= 0,
                        "PagedSelectiveSSM: num_processed_tokens[",
                        sequence,
                        "] must be non-negative, got ",
                        processed_tokens,
                        ".");

        const auto token_count = static_cast<uint64_t>(token_end - token_begin);
        if (token_count == 0) {
            continue;
        }

        const auto block_begin = static_cast<uint64_t>(block_begin_value);
        const auto block_end = static_cast<uint64_t>(block_end_value);
        OPENVINO_ASSERT(block_end > block_begin,
                        "PagedSelectiveSSM: non-empty sequence ",
                        sequence,
                        " requires a read block.");

        const auto interval = cache_intervals[sequence];
        if (interval <= 0) {
            continue;
        }
        const auto cache =
            PagedCacheSchedule::make(static_cast<int64_t>(interval), static_cast<uint64_t>(processed_tokens));
        OPENVINO_ASSERT(token_count <= std::numeric_limits<uint64_t>::max() - cache.offset,
                        "PagedSelectiveSSM: token count overflow at sequence ",
                        sequence,
                        ".");
        const auto write_count = cache.snapshot_count(token_count);
        const auto available_writes = block_end - block_begin - 1;
        OPENVINO_ASSERT(available_writes >= write_count,
                        "PagedSelectiveSSM: sequence ",
                        sequence,
                        " requires ",
                        write_count,
                        " writable logical blocks after the read block, got ",
                        available_writes,
                        ".");

        for (uint64_t slot = 1; slot <= write_count; ++slot) {
            const auto logical = static_cast<size_t>(block_begin + slot);
            const auto physical = checked_block_index(block_indices[logical], shape.physical_block_count, logical);
            OPENVINO_ASSERT(block_owners[physical] == -1,
                            "PagedSelectiveSSM: physical block ",
                            physical,
                            " is written more than once (previous sequence ",
                            block_owners[physical],
                            ", current sequence ",
                            sequence,
                            ").");
            block_owners[physical] = static_cast<int32_t>(sequence);
        }
    }

    for (size_t sequence = 0; sequence < shape.sequence_count; ++sequence) {
        if (subsequence_begins[sequence + 1] == subsequence_begins[sequence]) {
            continue;
        }
        const auto logical = static_cast<size_t>(block_indices_begins[sequence]);
        const auto physical = checked_block_index(block_indices[logical], shape.physical_block_count, logical);
        const auto owner = block_owners[physical];
        bool aliases_first_write = false;
        if (owner == static_cast<int32_t>(sequence)) {
            const auto first_write_logical = logical + 1;
            const auto first_write_physical = checked_block_index(block_indices[first_write_logical],
                                                                  shape.physical_block_count,
                                                                  first_write_logical);
            aliases_first_write = first_write_physical == physical;
        }
        OPENVINO_ASSERT(owner == -1 || aliases_first_write,
                        "PagedSelectiveSSM: sequence ",
                        sequence,
                        " reads physical block ",
                        physical,
                        " while sequence ",
                        owner,
                        " writes it; a read may alias only the same sequence's first write.");
    }
}

}  // namespace

void validate_paged_selective_ssm_jit_metadata(const PagedSelectiveSSMJitRuntimeArgs& args) {
    if (args.index_precision == ov::element::i32) {
        validate<int32_t>(args);
    } else if (args.index_precision == ov::element::i64) {
        validate<int64_t>(args);
    } else {
        OPENVINO_THROW("PagedSelectiveSSM JIT supports only i32/i64 metadata, got ", args.index_precision, ".");
    }
}

}  // namespace ov::intel_cpu::kernel
