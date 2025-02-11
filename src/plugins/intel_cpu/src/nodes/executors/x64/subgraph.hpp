// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/subgraph.hpp"

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
                 const std::vector<MemoryPtr>& inMemPtrs,
                 const std::vector<MemoryPtr>& outMemPtrs) override;

protected:
    std::vector<MemoryPtr> separately_repack_inputs(const dnnl::stream& strm, const std::vector<MemoryPtr>& srcMemPtrs);
    void in_parallel_repack_inputs(const std::vector<MemoryPtr>& inMemPtrs,
                                   const std::vector<size_t>& indexes,
                                   int ithr,
                                   jit_snippets_call_args& call_args);

    inline void* get_external_scratchpad_ptr(size_t ithr, size_t idx) const {
        if (m_repacked_inputs.empty()) {
            return nullptr;
        }

        uint8_t* data_ptr = m_buffer_scratchpad->getDataAs<uint8_t>() + m_internal_buffer_size;
        for (const auto& p : m_repacked_inputs) {
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
    std::vector<std::vector<size_t>> m_repacked_offsets_by_threads = {};
    std::unordered_map<size_t, RepackedInput> m_repacked_inputs = {};

    std::function<void(const std::vector<size_t>&, const std::vector<size_t>&, size_t&)> init_offset = {};

    using RepackingImplType = CPURuntimeConfig::RepackingImplType;
    const RepackingImplType& get_repacking_impl_type() const {
        return m_repacking_impl_type;
    }

    inline void clean_repacked_offsets(size_t ithr) {
        m_repacked_offsets_by_threads[ithr].assign(m_repacked_inputs.size(), std::numeric_limits<size_t>::max());
    }

#ifdef SNIPPETS_DEBUG_CAPS
    bool enabled_segfault_detector = false;
    inline void segfault_detector();
#endif

private:
    RepackingImplType m_repacking_impl_type = RepackingImplType::NONE;
};

class SubgraphStaticExecutor : public SubgraphExecutor, public SubgraphStaticBaseExecutor {
public:
    template <typename T, typename... Args>
    SubgraphStaticExecutor(T&& first, Args&&... rest)
        : SubgraphExecutor(std::forward<T>(first), std::forward<Args>(rest)...),
          SubgraphStaticBaseExecutor() {}

    void exec_impl(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

class SubgraphDynamicSpecializedExecutor : public SubgraphExecutor, public SubgraphDynamicSpecializedBaseExecutor {
public:
    template <typename T, typename... Args>
    SubgraphDynamicSpecializedExecutor(T&& first, Args&&... rest)
        : SubgraphExecutor(std::forward<T>(first), std::forward<Args>(rest)...),
          SubgraphDynamicSpecializedBaseExecutor(std::forward<T>(first)) {}

    void exec_impl(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

}  // namespace ov::intel_cpu
