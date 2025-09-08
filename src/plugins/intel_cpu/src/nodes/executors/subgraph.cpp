// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/subgraph.hpp"

#include <algorithm>
#include <common/utils.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <set>
#include <utility>
#include <vector>

#include "cache/multi_cache.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu_memory.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "snippets/generator.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu {

bool operator==(const SubgraphAttrs& lhs, const SubgraphAttrs& rhs) {
    if (&lhs == &rhs) {
        return true;
    }
    if (lhs.bodyHash != rhs.bodyHash) {
        return false;
    }
    if (lhs.inMemOrders.size() != rhs.inMemOrders.size() || lhs.inMemPrecs.size() != rhs.inMemPrecs.size()) {
        return false;
    }
    if (lhs.outMemOrders.size() != rhs.outMemOrders.size() || lhs.outMemPrecs.size() != rhs.outMemPrecs.size()) {
        return false;
    }
    if (lhs.inMemOrders != rhs.inMemOrders || lhs.inMemPrecs != rhs.inMemPrecs) {
        return false;
    }
    return lhs.outMemOrders == rhs.outMemOrders && lhs.outMemPrecs == rhs.outMemPrecs;
}

size_t get_attr_hash(size_t seed, const std::shared_ptr<SubgraphAttrs>& attrs) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    for (const auto& order : attrs->inMemOrders) {
        seed = get_vector_hash(seed, order);
    }
    for (const auto& prec : attrs->inMemPrecs) {
        seed = hash_combine(seed, prec.hash());
    }

    for (const auto& order : attrs->outMemOrders) {
        seed = get_vector_hash(seed, order);
    }
    for (const auto& prec : attrs->outMemPrecs) {
        seed = hash_combine(seed, prec.hash());
    }

    seed = hash_combine(seed, attrs->bodyHash);
    return seed;
}

SubgraphCodeGenerator::SubgraphCodeGenerator(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                             const std::shared_ptr<CPURuntimeConfig>& config,
                                             const std::set<size_t>& external_ptrs_idces) {
    OPENVINO_ASSERT(snippet_attrs, "Subgraph attributes are empty!");
    OPENVINO_ASSERT(config, "Runtime Config is empty!");

    jit_snippets_compile_args jcp;
    jcp.data_offsets.reserve(config->io_data_offsets.size());
    for (size_t i = 0; i < config->io_data_offsets.size(); ++i) {
        // Note: external ptrs must be ignored by the kernel during compilation
        if (!external_ptrs_idces.count(i)) {
            jcp.data_offsets.push_back(config->io_data_offsets[i]);
        }
    }
    SubgraphBaseExecutor::init_parallel_domain(config, jcp.exec_domain);
    schedule =
        std::make_shared<ov::snippets::Schedule>(snippet_attrs->snippet->generate(reinterpret_cast<const void*>(&jcp)));
}

SubgraphBaseExecutor::SubgraphBaseExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                                           [[maybe_unused]] const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                           const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                                           std::vector<ptrdiff_t> start_offset_in,
                                           std::vector<ptrdiff_t> start_offset_out,
                                           [[maybe_unused]] const BufferScratchpadAllocator& allocator,
                                           [[maybe_unused]] const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache)
    : m_schedule(snippet->get()),
      m_start_offset_in(std::move(start_offset_in)),
      m_start_offset_out(std::move(start_offset_out)) {
    OPENVINO_ASSERT(m_schedule, "Schedule is empty!");
    OPENVINO_ASSERT(snippet_config, "Runtime Config is empty!");
    init_parallel_domain(snippet_config, m_parallel_exec_domain);

    m_tensor_rank = snippet_config->tensor_rank;
    m_harness_work_amount = std::accumulate(m_parallel_exec_domain.cbegin(),
                                            m_parallel_exec_domain.cend(),
                                            static_cast<size_t>(1),
                                            std::multiplies<>());
    m_nthreads = std::min(parallel_get_max_threads(), static_cast<int>(m_harness_work_amount));

    m_buffer_scratchpad_size = snippet_config->buffer_scratchpad_size;
    OPENVINO_ASSERT(!ov::snippets::utils::is_dynamic_value(m_buffer_scratchpad_size),
                    "Undefined buffer scratchpad size!");
    m_internal_buffer_size = static_cast<size_t>(m_nthreads) * m_buffer_scratchpad_size;
}

void SubgraphBaseExecutor::init_parallel_domain(const std::vector<size_t>& master_shape,
                                                size_t tensor_rank,
                                                size_t tile_rank,
                                                std::vector<size_t>& domain) {
    domain.resize(tensor_rank, 1);
    std::fill(domain.begin(), domain.end(), 1);
    std::copy(master_shape.cbegin(),
              master_shape.cbegin() + (master_shape.size() - tile_rank),
              domain.begin() + (tensor_rank - master_shape.size()));
}

void SubgraphBaseExecutor::init_parallel_domain(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                                                std::vector<size_t>& domain) {
    init_parallel_domain(snippet_config->master_shape, snippet_config->tensor_rank, snippet_config->tile_rank, domain);
}

void SubgraphBaseExecutor::parallel_for6d(const initializer_functor& initializer, const call_functor& caller) {
    const auto& dom = m_parallel_exec_domain;

    parallel_nt_static(m_nthreads, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        initializer(call_args, ithr);

        size_t start = 0;
        size_t end = 0;
        splitter(m_harness_work_amount, nthr, ithr, start, end);

        std::vector<size_t> indexes{0, 0, 0, 0, 0};
        parallel_it_init(start,
                         indexes[0],
                         dom[0],
                         indexes[1],
                         dom[1],
                         indexes[2],
                         dom[2],
                         indexes[3],
                         dom[3],
                         indexes[4],
                         dom[4]);
        for (size_t iwork = start; iwork < end; ++iwork) {
            caller(call_args, indexes, ithr);
            parallel_it_step(indexes[0],
                             dom[0],
                             indexes[1],
                             dom[1],
                             indexes[2],
                             dom[2],
                             indexes[3],
                             dom[3],
                             indexes[4],
                             dom[4]);
        }
    });
}

void SubgraphBaseExecutor::parallel_forNd(const initializer_functor& initializer, const call_functor& caller) {
    const auto& dom = m_parallel_exec_domain;

    parallel_nt_static(m_nthreads, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        initializer(call_args, ithr);

        size_t start = 0;
        size_t end = 0;
        splitter(m_harness_work_amount, nthr, ithr, start, end);

        std::vector<size_t> indexes(dom.size() - 1, 0);
        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = static_cast<ptrdiff_t>(dom.size()) - 2; j >= 0; j--) {
                indexes[j] = tmp % dom[j];
                tmp /= dom[j];
            }

            caller(call_args, indexes, ithr);
        }
    });
}

void SubgraphBaseExecutor::execute([[maybe_unused]] const dnnl::stream& strm,
                                   const std::vector<MemoryPtr>& inMemPtrs,
                                   const std::vector<MemoryPtr>& outMemPtrs) {
    exec_impl(inMemPtrs, outMemPtrs);
}

}  // namespace ov::intel_cpu
