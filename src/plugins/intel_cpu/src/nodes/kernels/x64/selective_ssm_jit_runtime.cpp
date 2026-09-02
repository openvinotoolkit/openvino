// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_jit_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/selective_ssm.hpp"
#include "nodes/kernels/x64/jit_kernel_base.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_kernel.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_metadata.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/util/math_util.hpp"

namespace ov::intel_cpu::kernel {
namespace {

bool should_reuse_state_cache_as_fp32_working_buffer() {
    // Wide AVX-512 kernels benefit from private scratch; narrower kernels are limited by the extra state copies.
    return !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);
}

template <typename Destination, typename Source>
void copy_convert(Destination* destination, const Source* source, size_t count) {
    if constexpr (std::is_same_v<Destination, Source>) {
        if (destination != source) {
            std::memcpy(destination, source, count * sizeof(Destination));
        }
    } else {
        ov::Extensions::Cpu::XARCH::cvt_copy(destination, const_cast<Source*>(source), 1, count, count, count);
    }
}

void run_recurrence_kernel(const JitKernelBase& kernel,
                           const void* input_state,
                           void* output_state,
                           const float* input_projection,
                           const float* output_projection,
                           const void* input,
                           void* output,
                           float decay,
                           float time_step,
                           size_t row_count) {
    const jit_selective_ssm_call_args args{
        input_state,
        input_projection,
        output_projection,
        input,
        output,
        decay,
        time_step,
        row_count,
        output_state,
    };
    kernel(&args);
}

struct StateWorkLayout {
    size_t container_stride;
    size_t head_stride;
    size_t scratch_stride;
    size_t heads_per_group;
    size_t head_dim_tile_count;
};

StateWorkLayout make_state_work_layout(size_t num_heads,
                                       size_t num_groups,
                                       size_t head_dim,
                                       size_t state_size,
                                       size_t head_dim_tile,
                                       const char* container_name) {
    return {
        node::kernel::checked_size_product({num_heads, head_dim, state_size}, container_name),
        node::kernel::checked_size_product({head_dim, state_size}, "JIT recurrent state head"),
        node::kernel::checked_size_product({head_dim_tile, state_size}, "JIT state scratch"),
        num_heads / num_groups,
        ov::util::ceil_div(head_dim, head_dim_tile),
    };
}

struct TensorCursor {
    size_t token_head;
    size_t projection;
    size_t input;

    void advance(size_t num_heads, size_t projection_stride, size_t input_stride) {
        token_head += num_heads;
        projection += projection_stride;
        input += input_stride;
    }
};

struct CacheSchedule {
    bool enabled;
    uint64_t interval;
    uint64_t offset;

    [[nodiscard]] uint64_t cached_tokens(size_t processed_tokens) const {
        return offset + processed_tokens;
    }

    [[nodiscard]] bool should_store(uint64_t token_count, bool is_last) const {
        return enabled && (token_count % interval == 0 || is_last);
    }

    [[nodiscard]] size_t write_slot(uint64_t token_count) const {
        return 1 + (token_count - 1) / interval;
    }

    [[nodiscard]] size_t write_count(size_t processed_tokens) const {
        return enabled ? write_slot(cached_tokens(processed_tokens)) : 0;
    }
};

template <typename Data>
void run_selective_ssm(const Data* state_decay_rates,
                       const Data* time_steps,
                       const float* input_projections,
                       const Data* input,
                       const float* output_projections,
                       const Data* initial_state,
                       Data* output,
                       Data* final_state,
                       const node::kernel::SelectiveSSMShape& shape,
                       float* state_scratch,
                       size_t head_dim_tile,
                       const CpuParallelPtr& cpu_parallel,
                       const JitKernelBase& fp32_state_kernel,
                       const JitKernelBase* direct_state_kernel) {
    const auto batch_state_stride =
        node::kernel::checked_size_product({shape.num_heads, shape.head_dim, shape.state_size},
                                           "JIT recurrent state batch");
    const auto head_state_stride =
        node::kernel::checked_size_product({shape.head_dim, shape.state_size}, "JIT recurrent state head");
    const auto scratch_stride =
        node::kernel::checked_size_product({head_dim_tile, shape.state_size}, "JIT state scratch");
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto head_dim_tile_count = ov::util::ceil_div(shape.head_dim, head_dim_tile);
    const auto projection_stride = shape.num_groups * shape.state_size;
    const auto input_stride = shape.num_heads * shape.head_dim;

    cpu_parallel->parallel_for3d(
        shape.batch_size,
        shape.num_heads,
        head_dim_tile_count,
        [&](size_t batch, size_t head, size_t tile) {
            const auto head_dim_begin = tile * head_dim_tile;
            const auto head_dim_count = std::min(head_dim_tile, shape.head_dim - head_dim_begin);
            const auto projection_group = head / heads_per_group;
            const auto state_offset =
                batch * batch_state_stride + head * head_state_stride + head_dim_begin * shape.state_size;
            const auto state_elements = head_dim_count * shape.state_size;
            const auto state_decay_rate = static_cast<float>(state_decay_rates[head]);
            auto token_head_offset = (batch * shape.sequence_length) * shape.num_heads + head;
            auto projection_offset =
                ((batch * shape.sequence_length) * shape.num_groups + projection_group) * shape.state_size;
            auto input_offset = token_head_offset * shape.head_dim + head_dim_begin;

            if (shape.sequence_length == 1 && direct_state_kernel != nullptr) {
                const auto time_step = static_cast<float>(time_steps[token_head_offset]);
                run_recurrence_kernel(*direct_state_kernel,
                                      initial_state + state_offset,
                                      final_state + state_offset,
                                      input_projections + projection_offset,
                                      output_projections + projection_offset,
                                      input + input_offset,
                                      output + input_offset,
                                      std::exp(state_decay_rate * time_step),
                                      time_step,
                                      head_dim_count);
                return;
            }

            float* local_state = nullptr;
            if constexpr (std::is_same_v<Data, float>) {
                local_state = final_state + state_offset;
            } else {
                local_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * scratch_stride;
            }
            copy_convert(local_state, initial_state + state_offset, state_elements);

            for (size_t token = 0; token < shape.sequence_length; ++token) {
                const auto time_step = static_cast<float>(time_steps[token_head_offset]);
                run_recurrence_kernel(fp32_state_kernel,
                                      local_state,
                                      local_state,
                                      input_projections + projection_offset,
                                      output_projections + projection_offset,
                                      input + input_offset,
                                      output + input_offset,
                                      std::exp(state_decay_rate * time_step),
                                      time_step,
                                      head_dim_count);
                token_head_offset += shape.num_heads;
                projection_offset += projection_stride;
                input_offset += input_stride;
            }

            if constexpr (!std::is_same_v<Data, float>) {
                copy_convert(final_state + state_offset, local_state, state_elements);
            }
        });
}

template <typename Data>
void dispatch_selective_ssm(const SelectiveSSMJitRuntimeArgs& args) {
    run_selective_ssm(static_cast<const Data*>(args.state_decay_rates),
                      static_cast<const Data*>(args.time_steps),
                      args.input_projections,
                      static_cast<const Data*>(args.input),
                      args.output_projections,
                      static_cast<const Data*>(args.initial_state),
                      static_cast<Data*>(args.output),
                      static_cast<Data*>(args.final_state),
                      args.shape,
                      args.state_scratch,
                      args.head_dim_tile,
                      args.cpu_parallel,
                      *args.fp32_state_kernel,
                      args.direct_state_kernel);
}

template <typename Data, typename Index>
void run_paged_selective_ssm(const PagedSelectiveSSMJitRuntimeArgs& args) {
    const auto* state_decay_rates = static_cast<const Data*>(args.state_decay_rates);
    const auto* time_steps = static_cast<const Data*>(args.time_steps);
    const auto* input = static_cast<const Data*>(args.input);
    auto* state_cache = static_cast<Data*>(args.state_cache);
    const auto* subsequence_begins = static_cast<const Index*>(args.subsequence_begins);
    const auto* block_indices = static_cast<const Index*>(args.block_indices);
    const auto* block_indices_begins = static_cast<const Index*>(args.block_indices_begins);
    const auto* num_processed_tokens = static_cast<const Index*>(args.num_processed_tokens);
    const auto* cache_intervals = static_cast<const Index*>(args.cache_intervals);
    const auto* input_projections = args.input_projections;
    const auto* output_projections = args.output_projections;
    const auto* fp32_state_kernel = args.fp32_state_kernel;
    const auto* direct_state_kernel = args.direct_state_kernel;
    const auto* no_state_store_kernel = args.no_state_store_kernel;
    auto* state_scratch = args.state_scratch;
    auto* output = static_cast<Data*>(args.output);
    const auto& shape = args.shape;
    const auto head_dim_tile = args.head_dim_tile;

    const auto layout = make_state_work_layout(shape.num_heads,
                                               shape.num_groups,
                                               shape.head_dim,
                                               shape.state_size,
                                               head_dim_tile,
                                               "JIT state block");
    const auto projection_stride = shape.num_groups * shape.state_size;
    const auto input_stride = shape.num_heads * shape.head_dim;

    args.cpu_parallel->parallel_for3d(
        shape.sequence_count,
        shape.num_heads,
        layout.head_dim_tile_count,
        [&](size_t sequence, size_t head, size_t tile) {
            const auto token_begin = static_cast<size_t>(subsequence_begins[sequence]);
            const auto token_end = static_cast<size_t>(subsequence_begins[sequence + 1]);
            if (token_begin == token_end) {
                return;
            }

            const auto head_dim_begin = tile * head_dim_tile;
            const auto head_dim_count = std::min(head_dim_tile, shape.head_dim - head_dim_begin);
            const auto projection_group = head / layout.heads_per_group;
            const auto logical_block_begin = static_cast<size_t>(block_indices_begins[sequence]);
            const auto read_block = static_cast<size_t>(block_indices[logical_block_begin]);
            const auto state_offset = head * layout.head_stride + head_dim_begin * shape.state_size;
            const auto* initial_state = state_cache + read_block * layout.container_stride + state_offset;
            const auto state_decay_rate = static_cast<float>(state_decay_rates[head]);
            const auto interval = static_cast<int64_t>(cache_intervals[sequence]);
            const CacheSchedule cache{
                interval > 0,
                interval > 0 ? static_cast<uint64_t>(interval) : uint64_t{1},
                interval > 0 ? static_cast<uint64_t>(static_cast<int64_t>(num_processed_tokens[sequence])) %
                                   static_cast<uint64_t>(interval)
                             : uint64_t{0},
            };
            TensorCursor cursor{
                token_begin * shape.num_heads + head,
                (token_begin * shape.num_groups + projection_group) * shape.state_size,
                (token_begin * shape.num_heads + head) * shape.head_dim + head_dim_begin,
            };
            const auto state_elements = head_dim_count * shape.state_size;

            if (token_end == token_begin + 1) {
                const auto time_step = static_cast<float>(time_steps[cursor.token_head]);
                if (cache.enabled && direct_state_kernel != nullptr) {
                    const auto token_count = cache.cached_tokens(1);
                    const auto write_block =
                        static_cast<size_t>(block_indices[logical_block_begin + cache.write_slot(token_count)]);
                    auto* snapshot = state_cache + write_block * layout.container_stride + state_offset;
                    run_recurrence_kernel(*direct_state_kernel,
                                          initial_state,
                                          snapshot,
                                          input_projections + cursor.projection,
                                          output_projections + cursor.projection,
                                          input + cursor.input,
                                          output + cursor.input,
                                          std::exp(state_decay_rate * time_step),
                                          time_step,
                                          head_dim_count);
                    return;
                }
                if (!cache.enabled && no_state_store_kernel != nullptr) {
                    run_recurrence_kernel(*no_state_store_kernel,
                                          initial_state,
                                          nullptr,
                                          input_projections + cursor.projection,
                                          output_projections + cursor.projection,
                                          input + cursor.input,
                                          output + cursor.input,
                                          std::exp(state_decay_rate * time_step),
                                          time_step,
                                          head_dim_count);
                    return;
                }
            }

            auto* local_state = state_scratch + static_cast<size_t>(parallel_get_thread_num()) * layout.scratch_stride;
            if constexpr (std::is_same_v<Data, float>) {
                const auto snapshot_count = cache.write_count(token_end - token_begin);
                // A single f32 snapshot can hold the working state, avoiding the scratch buffer and final copy.
                if (snapshot_count == 1 && should_reuse_state_cache_as_fp32_working_buffer()) {
                    const auto write_block = static_cast<size_t>(block_indices[logical_block_begin + snapshot_count]);
                    local_state = state_cache + write_block * layout.container_stride + state_offset;
                }
            }
            copy_convert(local_state, initial_state, state_elements);

            for (size_t token = token_begin; token < token_end; ++token) {
                const auto time_step = static_cast<float>(time_steps[cursor.token_head]);
                run_recurrence_kernel(*fp32_state_kernel,
                                      local_state,
                                      local_state,
                                      input_projections + cursor.projection,
                                      output_projections + cursor.projection,
                                      input + cursor.input,
                                      output + cursor.input,
                                      std::exp(state_decay_rate * time_step),
                                      time_step,
                                      head_dim_count);

                const auto processed_tokens = (token - token_begin) + 1;
                const auto token_count = cache.cached_tokens(processed_tokens);
                const bool is_last = token + 1 == token_end;
                if (cache.should_store(token_count, is_last)) {
                    const auto write_block =
                        static_cast<size_t>(block_indices[logical_block_begin + cache.write_slot(token_count)]);
                    auto* snapshot = state_cache + write_block * layout.container_stride + state_offset;
                    copy_convert(snapshot, local_state, state_elements);
                }

                cursor.advance(shape.num_heads, projection_stride, input_stride);
            }
        });
}

template <typename Data>
void dispatch_paged_indices(const PagedSelectiveSSMJitRuntimeArgs& args) {
    if (args.index_precision == ov::element::i32) {
        run_paged_selective_ssm<Data, int32_t>(args);
    } else if (args.index_precision == ov::element::i64) {
        run_paged_selective_ssm<Data, int64_t>(args);
    } else {
        OPENVINO_THROW("PagedSelectiveSSM JIT supports only i32/i64 metadata, got ", args.index_precision, ".");
    }
}

}  // namespace

void selective_ssm_jit(const SelectiveSSMJitRuntimeArgs& args) {
    OPENVINO_ASSERT(args.head_dim_tile > 0 && args.state_scratch != nullptr);
    OPENVINO_ASSERT(args.cpu_parallel != nullptr, "SelectiveSSM JIT requires a CPU parallel executor.");
    OPENVINO_ASSERT(args.fp32_state_kernel != nullptr, "SelectiveSSM JIT kernel is not initialized.");

    if (args.data_precision == ov::element::f32) {
        dispatch_selective_ssm<float>(args);
    } else if (args.data_precision == ov::element::f16) {
        dispatch_selective_ssm<ov::float16>(args);
    } else if (args.data_precision == ov::element::bf16) {
        dispatch_selective_ssm<ov::bfloat16>(args);
    } else {
        OPENVINO_THROW("SelectiveSSM JIT supports only f32/f16/bf16, got ", args.data_precision, ".");
    }
}

void paged_selective_ssm_jit(const PagedSelectiveSSMJitRuntimeArgs& args) {
    OPENVINO_ASSERT(args.head_dim_tile > 0 && args.state_scratch != nullptr);
    OPENVINO_ASSERT(args.cpu_parallel != nullptr, "PagedSelectiveSSM JIT requires a CPU parallel executor.");
    OPENVINO_ASSERT(args.fp32_state_kernel != nullptr, "PagedSelectiveSSM JIT kernel is not initialized.");
    validate_paged_selective_ssm_jit_metadata(args);

    if (args.data_precision == ov::element::f32) {
        dispatch_paged_indices<float>(args);
    } else if (args.data_precision == ov::element::f16) {
        dispatch_paged_indices<ov::float16>(args);
    } else if (args.data_precision == ov::element::bf16) {
        dispatch_paged_indices<ov::bfloat16>(args);
    } else {
        OPENVINO_THROW("PagedSelectiveSSM JIT supports only f32/f16/bf16, got ", args.data_precision, ".");
    }
}

}  // namespace ov::intel_cpu::kernel
