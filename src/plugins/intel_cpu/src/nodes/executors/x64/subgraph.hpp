// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <vector>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/input_repacker.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "graph_context.h"
#include "nodes/executors/subgraph.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

class SubgraphExecutor : public SubgraphBaseExecutor {
public:
    SubgraphExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                     const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                     const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                     const std::vector<ptrdiff_t>& start_offset_in,
                     const std::vector<ptrdiff_t>& start_offset_out,
                     const BufferScratchpadAllocator& allocator,
                     const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache);

    void execute(const dnnl::stream& strm,
                 const std::vector<MemoryPtr>& in_mem_ptrs,
                 const std::vector<MemoryPtr>& out_mem_ptrs) override;

protected:
    static void separately_repack_input(const MemoryPtr& src_mem_ptr,
                                        const MemoryPtr& dst_mem_ptr,
                                        const ov::intel_cpu::InputRepacker& input_repacker,
                                        size_t tensor_rank);

    std::vector<MemoryPtr> separately_repack_inputs(const dnnl::stream& strm,
                                                    const std::vector<MemoryPtr>& src_mem_ptrs);
    void in_parallel_repack_inputs(const std::vector<MemoryPtr>& in_mem_ptrs,
                                   const std::vector<size_t>& indexes,
                                   int ithr,
                                   jit_snippets_call_args& call_args);

    void* get_external_scratchpad_ptr(size_t ithr, size_t idx) const {
        if (m_input_repackers.empty()) {
            return nullptr;
        }

        uint8_t* data_ptr = m_buffer_scratchpad->getDataAs<uint8_t>() + m_internal_buffer_size;
        for (const auto& p : m_input_repackers) {
            const auto& desc = p.second.desc();
            const auto size = desc->getCurrentMemSize();
            if (p.first == idx) {
                return data_ptr + ithr * size;
            }
            data_ptr += m_nthreads * size;
        }
        OPENVINO_THROW("External buffer pointer has not been found");
    }

    // [ Thread Index -> Index of input with repacking data - > last repacked src_offset ]
    std::vector<std::vector<size_t>> m_repacked_offsets_by_threads;
    InputRepackerMap m_input_repackers;

    std::function<void(const std::vector<size_t>&, const std::vector<size_t>&, size_t&)> init_offset;

    using RepackingImplType = CPURuntimeConfig::RepackingImplType;
    const RepackingImplType& get_repacking_impl_type() const {
        return m_repacking_impl_type;
    }

    void clean_repacked_offsets(size_t ithr) {
        m_repacked_offsets_by_threads[ithr].assign(m_input_repackers.size(), std::numeric_limits<size_t>::max());
    }

#ifdef SNIPPETS_DEBUG_CAPS
    bool enabled_segfault_detector = false;
    inline void segfault_detector() const;
#endif

private:
    RepackingImplType m_repacking_impl_type = RepackingImplType::NONE;
};

class SubgraphStaticExecutor : public SubgraphExecutor, public SubgraphStaticBaseExecutor {
public:
    template <typename... Args>
    SubgraphStaticExecutor(const std::shared_ptr<ov::intel_cpu::CPURuntimeConfig>& config,
                           const std::set<size_t>& external_ptrs_idces,
                           size_t in_num,
                           Args&&... rest)
        : SubgraphExecutor(config, std::forward<Args>(rest)...),
          SubgraphStaticBaseExecutor(external_ptrs_idces, in_num) {}

    void exec_impl(const std::vector<MemoryPtr>& in_mem_ptrs, const std::vector<MemoryPtr>& out_mem_ptrs) override;
};

class SubgraphDynamicSpecializedExecutor : public SubgraphExecutor, public SubgraphDynamicSpecializedBaseExecutor {
public:
    template <typename... Args>
    SubgraphDynamicSpecializedExecutor(const std::shared_ptr<ov::intel_cpu::CPURuntimeConfig>& config,
                                       const std::set<size_t>& external_ptrs_idces,
                                       size_t in_num,
                                       Args&&... rest)
        : SubgraphExecutor(config, std::forward<Args>(rest)...),
          SubgraphDynamicSpecializedBaseExecutor(config, external_ptrs_idces, in_num) {}

    void exec_impl(const std::vector<MemoryPtr>& in_mem_ptrs, const std::vector<MemoryPtr>& out_mem_ptrs) override;
};

}  // namespace ov::intel_cpu
