// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/generator.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov::intel_cpu {

struct SubgraphAttrs {
    // Local copy of subgraph node for canonization & code generation
    std::shared_ptr<snippets::op::Subgraph> snippet;
    uint64_t bodyHash;
    std::vector<VectorDims> inMemOrders;
    std::vector<VectorDims> outMemOrders;
    std::vector<ov::element::Type> inMemPrecs;
    std::vector<ov::element::Type> outMemPrecs;
};
bool operator==(const SubgraphAttrs& lhs, const SubgraphAttrs& rhs);
size_t get_attr_hash(size_t seed, const std::shared_ptr<SubgraphAttrs>& attrs);

class SubgraphCodeGenerator {
public:
    SubgraphCodeGenerator(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                          const std::shared_ptr<CPURuntimeConfig>& config);

    [[nodiscard]] const std::shared_ptr<snippets::Schedule>& get() const {
        return schedule;
    }

private:
    std::shared_ptr<snippets::Schedule> schedule;
};

class SubgraphBaseExecutor {
public:
    using BufferScratchpadAllocator = std::function<MemoryPtr(size_t)>;

    SubgraphBaseExecutor() = default;
    SubgraphBaseExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                         const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                         const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                         std::vector<ptrdiff_t> start_offset_in,
                         std::vector<ptrdiff_t> start_offset_out,
                         const BufferScratchpadAllocator& allocator,
                         const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache);
    virtual ~SubgraphBaseExecutor() = default;

    virtual void execute(const dnnl::stream& strm,
                         const std::vector<MemoryPtr>& inMemPtrs,
                         const std::vector<MemoryPtr>& outMemPtrs);

    static void init_parallel_domain(const std::vector<size_t>& master_shape,
                                     size_t tensor_rank,
                                     size_t tile_rank,
                                     std::vector<size_t>& domain);
    static void init_parallel_domain(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                                     std::vector<size_t>& domain);

protected:
    virtual void exec_impl(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;

    using initializer_functor = std::function<void(jit_snippets_call_args&, size_t)>;
    using call_functor = std::function<void(jit_snippets_call_args&, const std::vector<size_t>&, size_t)>;

    virtual void parallel_for6d(const initializer_functor& initializer, const call_functor& caller);
    virtual void parallel_forNd(const initializer_functor& initializer, const call_functor& caller);

    inline void update_scratchpad_ptr(void*& scratchpad_ptr, size_t ithr) const {
        if (m_buffer_scratchpad_size > 0) {
            scratchpad_ptr = m_buffer_scratchpad->getDataAs<uint8_t>() + ithr * m_buffer_scratchpad_size;
        }
    }

    std::shared_ptr<snippets::Schedule> m_schedule;
    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> m_parallel_exec_domain = {};
    size_t m_harness_work_amount = 0;

    // Buffer scratchpad
    MemoryPtr m_buffer_scratchpad = nullptr;
    size_t m_buffer_scratchpad_size = 0;
    size_t m_internal_buffer_size = 0;
    size_t m_tensor_rank = 0;

    const size_t rank6D = 6;

    // Count of threads for parallel_nt
    int m_nthreads = 0;

    std::vector<ptrdiff_t> m_start_offset_in = {};
    std::vector<ptrdiff_t> m_start_offset_out = {};
};

// Class for Subgraphs with static shapes
class SubgraphStaticBaseExecutor {
public:
    SubgraphStaticBaseExecutor() = default;
    virtual ~SubgraphStaticBaseExecutor() = default;

protected:
    using kernel = void (*)(const void*, const void*);

    inline void init_call_args(jit_snippets_call_args& call_args,
                               const std::vector<MemoryPtr>& srcMemPtrs,
                               const std::vector<MemoryPtr>& dstMemPtrs,
                               const std::vector<ptrdiff_t>& start_offset_in,
                               const std::vector<ptrdiff_t>& start_offset_out,
                               size_t ithr) {
        for (size_t i = 0; i < srcMemPtrs.size(); i++) {
            call_args.src_ptrs[i] = srcMemPtrs[i]->getDataAs<const uint8_t>() + start_offset_in[i];
        }

        for (size_t i = 0; i < dstMemPtrs.size(); i++) {
            call_args.dst_ptrs[i] = dstMemPtrs[i]->getDataAs<uint8_t>() + start_offset_out[i];
        }
    }
};

// Specialized dynamic executor based on shape agnostic kernel for the specific input shapes
class SubgraphDynamicSpecializedBaseExecutor {
public:
    SubgraphDynamicSpecializedBaseExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config)
        : m_buffer_offsets(snippet_config->buffer_cluster_offsets),
          m_data_offsets(snippet_config->io_data_offsets),
          m_loop_args(snippet_config->loop_args) {
        m_reset_exec_table_state = snippet_config->kernel_executor_table->get_state_reset();
    }
    virtual ~SubgraphDynamicSpecializedBaseExecutor() = default;

protected:
    using dynamic_kernel = void (*)(const void*);

    inline void init_call_args(jit_snippets_call_args& call_args, size_t ithr) {
        call_args.register_loops(m_loop_args);
        std::copy(m_buffer_offsets.cbegin(), m_buffer_offsets.cend(), call_args.buffer_offsets);
    }

    inline void init_original_ptrs(const std::vector<MemoryPtr>& srcMemPtrs,
                                   const std::vector<MemoryPtr>& dstMemPtrs,
                                   std::vector<const uint8_t*>& src_ptrs,
                                   std::vector<uint8_t*>& dst_ptrs,
                                   const std::vector<ptrdiff_t>& start_offset_in,
                                   const std::vector<ptrdiff_t>& start_offset_out) {
        const auto in_num = srcMemPtrs.size();
        const auto out_num = dstMemPtrs.size();

        src_ptrs.resize(in_num, nullptr);
        dst_ptrs.resize(out_num, nullptr);

        for (size_t i = 0; i < in_num; i++) {
            src_ptrs[i] = srcMemPtrs[i]->getDataAs<const uint8_t>() + start_offset_in[i];
        }
        for (size_t i = 0; i < out_num; i++) {
            dst_ptrs[i] = dstMemPtrs[i]->getDataAs<uint8_t>() + start_offset_out[i];
        }
    }

    inline void update_ptrs(jit_snippets_call_args& call_args,
                            const std::vector<const uint8_t*>& src_ptrs,
                            const std::vector<uint8_t*>& dst_ptrs,
                            const std::vector<size_t>& indexes) const {
        for (size_t i = 0; i < src_ptrs.size(); i++) {
            auto i_ptr = src_ptrs[i];
            for (size_t j = 0; j < indexes.size(); j++) {
                i_ptr += m_data_offsets[i][j] * indexes[j];
            }
            call_args.src_ptrs[i] = i_ptr;
        }
        for (size_t i = 0; i < dst_ptrs.size(); i++) {
            auto i_ptr = dst_ptrs[i];
            for (size_t j = 0; j < indexes.size(); j++) {
                i_ptr += m_data_offsets[i + src_ptrs.size()][j] * indexes[j];
            }
            call_args.dst_ptrs[i] = i_ptr;
        }
    }

    std::vector<size_t> m_buffer_offsets = {};
    std::vector<std::vector<size_t>> m_data_offsets = {};
    std::vector<jit_snippets_call_args::loop_args_t> m_loop_args = {};
    std::function<void()> m_reset_exec_table_state;
};

}  // namespace ov::intel_cpu
