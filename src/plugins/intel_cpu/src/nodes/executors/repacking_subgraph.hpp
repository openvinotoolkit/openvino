// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <set>
#include <utility>
#include <vector>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/input_repacker.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/subgraph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

template <typename RepackingKernel>
class SubgraphRepackingExecutor : public SubgraphBaseExecutor {
public:
    SubgraphRepackingExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                              const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                              const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                              const std::vector<ptrdiff_t>& start_offset_in,
                              const std::vector<ptrdiff_t>& start_offset_out,
                              const BufferScratchpadAllocator& allocator,
                              const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache)
        : SubgraphBaseExecutor(snippet_config,
                               snippet_attrs,
                               snippet,
                               start_offset_in,
                               start_offset_out,
                               allocator,
                               kernel_cache),
          m_input_repackers(snippet_config->input_repackers),
          m_repacking_impl_type(snippet_config->repacking_impl_type) {
        auto external_buffer_size =
            std::accumulate(m_input_repackers.begin(),
                            m_input_repackers.end(),
                            static_cast<size_t>(0),
                            [](size_t sum, const std::pair<size_t, InputRepacker>& p) {
                                auto curr_mem_size = p.second.desc()->getCurrentMemSize();
                                OPENVINO_ASSERT(curr_mem_size != ov::intel_cpu::MemoryDesc::UNDEFINED_SIZE,
                                                "Current repacking buffer memory size is undefined");
                                return sum + curr_mem_size;
                            });

        if (get_repacking_impl_type() == RepackingImplType::IN_PARALLEL) {
            external_buffer_size *= m_nthreads;

            m_repacked_offsets_by_threads.resize(m_nthreads);
            for (size_t i = 0; i < m_repacked_offsets_by_threads.size(); ++i) {
                clean_repacked_offsets(i);
            }

            if (m_tensor_rank == rank6D) {
                init_offset =
                    [](const std::vector<size_t>& offsets, const std::vector<size_t>& indexes, size_t& offset) {
                        offset += offsets[0] * indexes[0] + offsets[1] * indexes[1] + offsets[2] * indexes[2] +
                                  offsets[3] * indexes[3];
                    };
            } else {
                init_offset =
                    [](const std::vector<size_t>& offsets, const std::vector<size_t>& indexes, size_t& offset) {
                        for (size_t j = 0; j < indexes.size(); j++) {
                            offset += offsets[j] * indexes[j];
                        }
                    };
            }
        }

        m_buffer_scratchpad = allocator(m_internal_buffer_size + external_buffer_size);
    }

    void execute(const dnnl::stream& strm,
                 const std::vector<MemoryPtr>& in_mem_ptrs,
                 const std::vector<MemoryPtr>& out_mem_ptrs) override {
        switch (get_repacking_impl_type()) {
        case RepackingImplType::SEPARATE:
            exec_impl(separately_repack_inputs(strm, in_mem_ptrs), out_mem_ptrs);
            return;
        case RepackingImplType::IN_PARALLEL:
        case RepackingImplType::NONE:
            exec_impl(in_mem_ptrs, out_mem_ptrs);
            return;
        default:
            OPENVINO_THROW("Uknown RepackingImplType");
        }
    }

protected:
    static void parallel4d_repacking(const RepackingKernel* ker,
                                     const VectorDims& dom,
                                     const VectorDims& in_str,
                                     const VectorDims& out_str,
                                     const uint8_t* const src,
                                     uint8_t* const dst) {  // NOLINT(readability-non-const-parameter)
        parallel_for4d(dom[0], dom[1], dom[2], dom[3], [&](size_t d0, size_t d1, size_t d2, size_t d3) {
            typename RepackingKernel::call_args args;
            args.src = src + d0 * in_str[0] + d1 * in_str[1] + d2 * in_str[2] + d3 * in_str[3];
            args.tr_src = dst + d0 * out_str[0] + d1 * out_str[1] + d2 * out_str[2] + d3 * out_str[3];
            (*ker)(&args);
        });
    }

    static void parallelNd_repacking(const RepackingKernel* ker,
                                     const VectorDims& dom,
                                     const VectorDims& in_str,
                                     const VectorDims& out_str,
                                     const uint8_t* const src,
                                     uint8_t* const dst) {  // NOLINT(readability-non-const-parameter)
        const size_t batch = std::accumulate(dom.rbegin() + 2, dom.rend(), 1LU, std::multiplies<>());
        parallel_nt_static(0, [&](const int ithr, const int nthr) {
            typename RepackingKernel::call_args args;
            size_t start = 0;
            size_t end = 0;
            splitter(batch, nthr, ithr, start, end);
            for (size_t iwork = start; iwork < end; ++iwork) {
                const uint8_t* src_u8 = src;
                uint8_t* dst_u8 = dst;
                size_t tmp = iwork;
                for (ptrdiff_t j = static_cast<ptrdiff_t>(dom.size()) - 3; j >= 0; j--) {
                    auto idx = tmp % dom[j];
                    tmp /= dom[j];

                    src_u8 += idx * in_str[j];
                    dst_u8 += idx * out_str[j];
                }
                args.src = src_u8;
                args.tr_src = dst_u8;
                (*ker)(&args);
            }
        });
    }

    static void separately_repack_input(const MemoryPtr& src_mem_ptr,
                                        const MemoryPtr& dst_mem_ptr,
                                        const ov::intel_cpu::InputRepacker& input_repacker,
                                        size_t tensor_rank) {
        auto get_offset = [](const BlockedMemoryDescPtr& desc) {
            return static_cast<ptrdiff_t>(desc->getOffsetPadding() * desc->getPrecision().size());
        };

        const auto* src_ptr =
            src_mem_ptr->getDataAs<const uint8_t>() + get_offset(src_mem_ptr->getDescWithType<BlockedMemoryDesc>());
        auto* dst_ptr =
            dst_mem_ptr->getDataAs<uint8_t>() + get_offset(dst_mem_ptr->getDescWithType<BlockedMemoryDesc>());

        VectorDims dom;
        const auto& shape = dst_mem_ptr->getShape().getDims();
        OPENVINO_ASSERT(shape.size() <= tensor_rank, "Unsupported shape rank of repacking data");
        init_parallel_domain(shape, tensor_rank, 2LU, dom);

        const auto& in_strides = input_repacker.in_offsets();
        const auto& out_strides = input_repacker.out_offsets();
        OPENVINO_ASSERT(all_of(tensor_rank, in_strides.size(), out_strides.size(), dom.size()),
                        "Unsupported shape rank of repacking data");

        const auto& kernel = input_repacker.template kernel<RepackingKernel>();
        if (tensor_rank == rank6D) {
            parallel4d_repacking(kernel.get(), dom, in_strides, out_strides, src_ptr, dst_ptr);
        } else {
            parallelNd_repacking(kernel.get(), dom, in_strides, out_strides, src_ptr, dst_ptr);
        }
    }

    std::vector<MemoryPtr> separately_repack_inputs(const dnnl::stream& strm,
                                                    const std::vector<MemoryPtr>& src_mem_ptrs) {
        auto reordered_in_ptrs = src_mem_ptrs;
        size_t offset = m_internal_buffer_size;
        for (const auto& [in_idx, input_repacker] : m_input_repackers) {
            const auto& desc = input_repacker.desc();
            const void* data_ptr = m_buffer_scratchpad->getDataAs<uint8_t>() + offset;

            OPENVINO_ASSERT(in_idx < src_mem_ptrs.size(), "Incorrect index of input repacked mem ptr");
            const auto& src_mem = src_mem_ptrs[in_idx];
            const auto& dst_mem = std::make_shared<Memory>(strm.get_engine(), desc, data_ptr, false);
            separately_repack_input(src_mem, dst_mem, input_repacker, m_tensor_rank);

            reordered_in_ptrs[in_idx] = dst_mem;
            offset += desc->getCurrentMemSize();
        }
        return reordered_in_ptrs;
    }

    void in_parallel_repack_inputs(const std::vector<MemoryPtr>& in_mem_ptrs,
                                   const std::vector<size_t>& indexes,
                                   int ithr,
                                   jit_snippets_call_args& call_args) {
        size_t repacked_offset_idx = 0;
        for (const auto& [in_idx, input_repacker] : m_input_repackers) {
            size_t src_offset = m_start_offset_in[in_idx];
            init_offset(input_repacker.in_offsets(), indexes, src_offset);

            auto* repacked_ptr = get_external_scratchpad_ptr(ithr, in_idx);

            auto& last_processed_src_offset = m_repacked_offsets_by_threads[ithr][repacked_offset_idx];
            if (src_offset != last_processed_src_offset) {
                typename RepackingKernel::call_args args;
                args.src = in_mem_ptrs[in_idx]->getDataAs<const uint8_t>() + src_offset;
                args.tr_src = repacked_ptr;
                (*input_repacker.template kernel<RepackingKernel>())(&args);

                last_processed_src_offset = src_offset;
            }

            call_args.src_ptrs[in_idx] = repacked_ptr;
            ++repacked_offset_idx;
        }
    }

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

    using RepackingImplType = CPURuntimeConfig::RepackingImplType;
    const RepackingImplType& get_repacking_impl_type() const {
        return m_repacking_impl_type;
    }

    void clean_repacked_offsets(size_t ithr) {
        m_repacked_offsets_by_threads[ithr].assign(m_input_repackers.size(), std::numeric_limits<size_t>::max());
    }

#ifdef SNIPPETS_DEBUG_CAPS
    virtual void segfault_detector() const {}
#endif

    std::vector<std::vector<size_t>> m_repacked_offsets_by_threads;
    InputRepackerMap m_input_repackers;
    std::function<void(const std::vector<size_t>&, const std::vector<size_t>&, size_t&)> init_offset;

private:
    RepackingImplType m_repacking_impl_type = RepackingImplType::NONE;
};

template <typename RepackingExecutor>
class SubgraphRepackingStaticExecutor : public RepackingExecutor, public SubgraphStaticBaseExecutor {
public:
    template <typename... Args>
    SubgraphRepackingStaticExecutor(const std::shared_ptr<ov::intel_cpu::CPURuntimeConfig>& config,
                                    const std::set<size_t>& external_ptrs_idces,
                                    size_t in_num,
                                    Args&&... rest)
        : RepackingExecutor(config, std::forward<Args>(rest)...),
          SubgraphStaticBaseExecutor(external_ptrs_idces, in_num) {}

private:
    void exec_impl(const std::vector<MemoryPtr>& in_mem_ptrs, const std::vector<MemoryPtr>& out_mem_ptrs) override {
        const auto& callable = this->m_schedule->template get_callable<typename SubgraphStaticBaseExecutor::kernel>();

        typename SubgraphBaseExecutor::initializer_functor initializer;
        typename SubgraphBaseExecutor::call_functor caller;

        switch (this->get_repacking_impl_type()) {
        case RepackingExecutor::RepackingImplType::IN_PARALLEL:
            initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
                this->init_call_args(call_args,
                                     in_mem_ptrs,
                                     out_mem_ptrs,
                                     this->m_start_offset_in,
                                     this->m_start_offset_out);
                this->update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
                this->clean_repacked_offsets(ithr);
            };
            caller = [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, size_t ithr) {
                this->in_parallel_repack_inputs(in_mem_ptrs, indexes, ithr, call_args);
                callable(&call_args, indexes.data());
            };
            break;
        case RepackingExecutor::RepackingImplType::SEPARATE:
        case RepackingExecutor::RepackingImplType::NONE:
            initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
                this->init_call_args(call_args,
                                     in_mem_ptrs,
                                     out_mem_ptrs,
                                     this->m_start_offset_in,
                                     this->m_start_offset_out);
                this->update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
            };
            caller = [&](jit_snippets_call_args& call_args,
                         const std::vector<size_t>& indexes,
                         [[maybe_unused]] size_t ithr) {
                callable(&call_args, indexes.data());
            };
            break;
        default:
            OPENVINO_THROW("Uknown RepackingImplType");
        }

#if defined(SNIPPETS_DEBUG_CAPS) && (defined(__linux__) || defined(__APPLE__))
        this->segfault_detector();
#endif

        if (this->m_parallel_exec_domain.size() == this->rank6D) {
            this->parallel_for6d(initializer, caller);
        } else {
            this->parallel_forNd(initializer, caller);
        }
    }
};

template <typename RepackingExecutor>
class SubgraphRepackingDynamicSpecializedExecutor : public RepackingExecutor,
                                                    public SubgraphDynamicSpecializedBaseExecutor {
public:
    template <typename... Args>
    SubgraphRepackingDynamicSpecializedExecutor(const std::shared_ptr<ov::intel_cpu::CPURuntimeConfig>& config,
                                                const std::set<size_t>& external_ptrs_idces,
                                                size_t in_num,
                                                Args&&... rest)
        : RepackingExecutor(config, std::forward<Args>(rest)...),
          SubgraphDynamicSpecializedBaseExecutor(config, external_ptrs_idces, in_num) {}

private:
    void exec_impl(const std::vector<MemoryPtr>& in_mem_ptrs, const std::vector<MemoryPtr>& out_mem_ptrs) override {
        const auto& callable =
            this->m_schedule->template get_callable<typename SubgraphDynamicSpecializedBaseExecutor::dynamic_kernel>();

        OPENVINO_ASSERT(this->m_data_offsets.size() == in_mem_ptrs.size() + out_mem_ptrs.size(),
                        "Incorrect data offset count!");
        OPENVINO_ASSERT(this->m_data_offsets.front().size() == this->m_parallel_exec_domain.size(),
                        "Data offsets with invalid ranks detected");

        this->m_reset_exec_table_state();

        std::vector<const uint8_t*> src_ptrs;
        std::vector<uint8_t*> dst_ptrs;
        this->init_original_ptrs(in_mem_ptrs,
                                 out_mem_ptrs,
                                 src_ptrs,
                                 dst_ptrs,
                                 this->m_start_offset_in,
                                 this->m_start_offset_out);

        typename SubgraphBaseExecutor::initializer_functor initializer;
        typename SubgraphBaseExecutor::call_functor caller;

        switch (this->get_repacking_impl_type()) {
        case RepackingExecutor::RepackingImplType::IN_PARALLEL:
            initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
                this->init_call_args(call_args);
                this->update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
                this->clean_repacked_offsets(ithr);
            };
            caller = [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, size_t ithr) {
                this->update_ptrs(call_args, src_ptrs, dst_ptrs, indexes);
                this->in_parallel_repack_inputs(in_mem_ptrs, indexes, ithr, call_args);
                callable(&call_args);
            };
            break;
        case RepackingExecutor::RepackingImplType::SEPARATE:
        case RepackingExecutor::RepackingImplType::NONE:
            initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
                this->init_call_args(call_args);
                this->update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
            };
            caller = [&](jit_snippets_call_args& call_args,
                         const std::vector<size_t>& indexes,
                         [[maybe_unused]] size_t ithr) {
                this->update_ptrs(call_args, src_ptrs, dst_ptrs, indexes);
                callable(&call_args);
            };
            break;
        default:
            OPENVINO_THROW("Uknown RepackingImplType");
        }

#if defined(SNIPPETS_DEBUG_CAPS) && (defined(__linux__) || defined(__APPLE__))
        this->segfault_detector();
#endif

        if (this->m_parallel_exec_domain.size() == this->rank6D) {
            this->parallel_for6d(initializer, caller);
        } else {
            this->parallel_forNd(initializer, caller);
        }
    }
};

}  // namespace ov::intel_cpu
