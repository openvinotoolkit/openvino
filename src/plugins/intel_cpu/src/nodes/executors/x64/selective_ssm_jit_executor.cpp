// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_jit_executor.hpp"

#include <algorithm>
#include <array>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/paged_selective_ssm_config.hpp"
#include "nodes/executors/selective_ssm_config.hpp"
#include "nodes/kernels/selective_ssm.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_config.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_kernel.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_runtime.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {
namespace {

constexpr size_t min_parallel_projection_elements = 4096;

struct SelectiveSSMJitKey {
    ov::element::Type data_precision;
    ov::element::Type state_precision;
    size_t state_size;
    kernel::jit_selective_ssm_state_mode state_mode;

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, data_precision.hash());
        seed = dnnl::impl::hash_combine(seed, state_precision.hash());
        seed = dnnl::impl::hash_combine(seed, state_size);
        return dnnl::impl::hash_combine(seed, static_cast<size_t>(state_mode));
    }

    bool operator==(const SelectiveSSMJitKey& rhs) const {
        return data_precision == rhs.data_precision && state_precision == rhs.state_precision &&
               state_size == rhs.state_size && state_mode == rhs.state_mode;
    }
};

bool has_jit_isa() {
    return mayiuse(dnnl::impl::cpu::x64::avx512_core) || mayiuse(dnnl::impl::cpu::x64::avx2);
}

bool is_supported_data_precision(const ov::element::Type& precision) {
    return any_of(precision, ov::element::f32, ov::element::f16, ov::element::bf16);
}

bool is_supported_index_precision(const ov::element::Type& precision) {
    return any_of(precision, ov::element::i32, ov::element::i64);
}

bool is_supported_state_size(size_t state_size) {
    return state_size > 0 && state_size <= kernel::max_selective_ssm_jit_state_size;
}

template <typename Config, typename Argument, size_t Size>
bool all_arguments_have_precision(const Config& config,
                                  const std::array<Argument, Size>& arguments,
                                  const ov::element::Type& precision) {
    return std::all_of(arguments.begin(), arguments.end(), [&](Argument argument) {
        return config.descs.at(argument)->getPrecision() == precision;
    });
}

std::shared_ptr<kernel::JitKernelBase> get_or_create_kernel(const ExecutorContext::CPtr& context,
                                                            const ov::element::Type& data_precision,
                                                            const ov::element::Type& state_precision,
                                                            size_t state_size,
                                                            kernel::jit_selective_ssm_state_mode state_mode) {
    const SelectiveSSMJitKey key{data_precision, state_precision, state_size, state_mode};
    auto builder = [](const SelectiveSSMJitKey& compile_key) {
        return kernel::create_selective_ssm_jit_kernel(compile_key.data_precision,
                                                       compile_key.state_size,
                                                       compile_key.state_precision,
                                                       compile_key.state_mode);
    };
    return context->getRuntimeCache()->getOrCreate(key, builder).first;
}

}  // namespace

SelectiveSSMJitExecutorBase::SelectiveSSMJitExecutorBase(ExecutorContext::CPtr context)
    : m_context(std::move(context)) {}

bool SelectiveSSMJitExecutorBase::configure_resources(const ResourceRequirements& requirements) {
    if (!is_supported_state_size(requirements.state_size) || requirements.head_dim_tile == 0) {
        return false;
    }
    if (m_kernels.ready(requirements.needs_no_state_store_kernel) && m_scratch != nullptr &&
        m_requirements == requirements) {
        return true;
    }

    KernelBundle kernels;
    kernels.fp32_state = get_or_create_kernel(m_context,
                                              requirements.data_precision,
                                              ov::element::f32,
                                              requirements.state_size,
                                              kernel::jit_selective_ssm_state_mode::in_place);
    if (kernels.fp32_state == nullptr) {
        return false;
    }
    kernels.direct_state = get_or_create_kernel(m_context,
                                                requirements.data_precision,
                                                requirements.data_precision,
                                                requirements.state_size,
                                                kernel::jit_selective_ssm_state_mode::separate);
    if (kernels.direct_state == nullptr) {
        return false;
    }
    if (requirements.needs_no_state_store_kernel) {
        kernels.no_state_store = get_or_create_kernel(m_context,
                                                      requirements.data_precision,
                                                      requirements.data_precision,
                                                      requirements.state_size,
                                                      kernel::jit_selective_ssm_state_mode::no_store);
        if (kernels.no_state_store == nullptr) {
            return false;
        }
    }
    const auto projection_scratch_elements =
        requirements.data_precision == ov::element::f32
            ? size_t{0}
            : node::kernel::checked_size_product({size_t{2}, requirements.projection_elements},
                                                 "JIT B/C projection scratch");
    const auto total_scratch_elements = node::kernel::checked_size_sum(
        {requirements.state_scratch_elements, projection_scratch_elements, requirements.metadata_scratch_elements},
        "JIT combined scratch");
    const auto descriptor =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{std::max(size_t{1}, total_scratch_elements)});
    auto scratch = m_context->getScratchPad()->createScratchPadMem(descriptor);
    if (scratch == nullptr) {
        return false;
    }

    m_kernels = std::move(kernels);
    m_scratch = std::move(scratch);
    m_requirements = requirements;
    return true;
}

std::pair<const float*, const float*> SelectiveSSMJitExecutorBase::prepare_projections(const void* B,
                                                                                       const void* C) const {
    if (m_requirements.data_precision == ov::element::f32) {
        return {static_cast<const float*>(B), static_cast<const float*>(C)};
    }

    auto* converted_B = m_scratch->getDataAs<float>() + m_requirements.state_scratch_elements;
    auto* converted_C = converted_B + m_requirements.projection_elements;
    const auto convert = [&](const void* source, float* destination) {
        if (m_requirements.projection_elements < min_parallel_projection_elements) {
            cpu_convert(source,
                        destination,
                        m_requirements.data_precision,
                        ov::element::f32,
                        m_requirements.projection_elements);
        } else {
            cpu_parallel_convert(source,
                                 destination,
                                 m_requirements.data_precision,
                                 ov::element::f32,
                                 m_requirements.projection_elements);
        }
    };
    convert(B, converted_B);
    convert(C, converted_C);
    return {converted_B, converted_C};
}

impl_desc_type SelectiveSSMJitExecutorBase::implType() const {
    return mayiuse(dnnl::impl::cpu::x64::avx512_core) ? impl_desc_type::jit_avx512 : impl_desc_type::jit_avx2;
}

bool SelectiveSSMJitExecutor::supports(const SelectiveSSMConfig& config) {
    if (!has_jit_isa()) {
        return false;
    }
    const auto precision = config.descs.at(ARG_SSM_A)->getPrecision();
    if (!is_supported_data_precision(precision)) {
        return false;
    }
    constexpr std::array arguments{
        ARG_SSM_DT,
        ARG_SSM_B,
        ARG_SSM_X,
        ARG_SSM_C,
        ARG_SSM_STATE,
        ARG_SSM_OUT,
        ARG_SSM_OUT_STATE,
    };
    return all_arguments_have_precision(config, arguments, precision);
}

bool SelectiveSSMJitExecutor::accepts_shape(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_SSM_X)->getDescPtr()->getShape();
    const auto& B_shape = memory.at(ARG_SSM_B)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_SSM_STATE)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || B_shape.isDynamic() || state_shape.isDynamic()) {
        return false;
    }
    const auto& x_dims = x_shape.getStaticDims();
    const auto& B_dims = B_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    return x_dims.size() == 4 && B_dims.size() == 4 && state_dims.size() == 4 &&
           is_supported_state_size(state_dims.back());
}

SelectiveSSMJitExecutor::SelectiveSSMJitExecutor([[maybe_unused]] const SelectiveSSMAttrs& attrs,
                                                 const MemoryArgs& memory,
                                                 ExecutorContext::CPtr context)
    : SelectiveSSMJitExecutorBase(std::move(context)) {
    OPENVINO_ASSERT(update(memory), "Failed to initialize SelectiveSSM JIT executor");
}

bool SelectiveSSMJitExecutor::update(const MemoryArgs& memory) {
    if (!accepts_shape(memory)) {
        return false;
    }
    const auto& x_dims = memory.at(ARG_SSM_X)->getStaticDims();
    const auto& B_dims = memory.at(ARG_SSM_B)->getStaticDims();
    const auto& state_dims = memory.at(ARG_SSM_STATE)->getStaticDims();
    const auto precision = memory.at(ARG_SSM_X)->getDescPtr()->getPrecision();
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto outer_work = node::kernel::checked_size_product({x_dims[0], x_dims[2]}, "JIT outer work items");
    const auto head_dim_tile = node::kernel::get_scratch_head_dim(x_dims[3], state_dims[3], outer_work, thread_count);
    const bool direct_decode = x_dims[1] == 1;
    const auto state_scratch_elements =
        precision == ov::element::f32 || direct_decode
            ? size_t{0}
            : node::kernel::checked_size_product({thread_count, head_dim_tile, state_dims[3]}, "JIT state scratch");
    const auto projection_elements =
        node::kernel::checked_size_product({B_dims[0], B_dims[1], B_dims[2], B_dims[3]}, "JIT B/C projection");
    return configure_resources({precision, state_dims[3], head_dim_tile, state_scratch_elements, projection_elements});
}

void SelectiveSSMJitExecutor::execute(const MemoryArgs& memory) {
    OPENVINO_ASSERT(m_kernels.ready(false) && m_scratch != nullptr,
                    "SelectiveSSM JIT executor resources are not initialized");
    const auto& x_dims = memory.at(ARG_SSM_X)->getStaticDims();
    const auto& B_dims = memory.at(ARG_SSM_B)->getStaticDims();
    const auto& state_dims = memory.at(ARG_SSM_STATE)->getStaticDims();
    const auto precision = memory.at(ARG_SSM_X)->getDescPtr()->getPrecision();
    const node::kernel::SelectiveSSMShape shape{x_dims[0], x_dims[1], x_dims[2], x_dims[3], B_dims[2], state_dims[3]};
    const auto [input_projections, output_projections] =
        prepare_projections(memory.at(ARG_SSM_B)->getData(), memory.at(ARG_SSM_C)->getData());
    kernel::SelectiveSSMJitRuntimeArgs args;
    args.state_decay_rates = memory.at(ARG_SSM_A)->getData();
    args.time_steps = memory.at(ARG_SSM_DT)->getData();
    args.input_projections = input_projections;
    args.input = memory.at(ARG_SSM_X)->getData();
    args.output_projections = output_projections;
    args.initial_state = memory.at(ARG_SSM_STATE)->getData();
    args.output = memory.at(ARG_SSM_OUT)->getData();
    args.final_state = memory.at(ARG_SSM_OUT_STATE)->getData();
    args.shape = shape;
    args.data_precision = precision;
    args.state_scratch = m_scratch->getDataAs<float>();
    args.head_dim_tile = m_requirements.head_dim_tile;
    args.cpu_parallel = m_context->getCpuParallel();
    args.fp32_state_kernel = m_kernels.fp32_state.get();
    args.direct_state_kernel = m_kernels.direct_state.get();
    kernel::selective_ssm_jit(args);
}

bool PagedSelectiveSSMJitExecutor::supports(const PagedSelectiveSSMConfig& config) {
    if (!has_jit_isa()) {
        return false;
    }
    const auto data_precision = config.descs.at(ARG_PAGED_SSM_A)->getPrecision();
    if (!is_supported_data_precision(data_precision)) {
        return false;
    }
    constexpr std::array data_arguments{
        ARG_PAGED_SSM_DT,
        ARG_PAGED_SSM_B,
        ARG_PAGED_SSM_X,
        ARG_PAGED_SSM_C,
        ARG_PAGED_SSM_STATE,
        ARG_PAGED_SSM_OUT,
    };
    if (!all_arguments_have_precision(config, data_arguments, data_precision)) {
        return false;
    }

    const auto index_precision = config.descs.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getPrecision();
    if (!is_supported_index_precision(index_precision)) {
        return false;
    }
    constexpr std::array index_arguments{
        ARG_PAGED_SSM_BLOCK_INDICES,
        ARG_PAGED_SSM_BLOCK_INDICES_BEGINS,
        ARG_PAGED_SSM_NUM_PROCESSED_TOKENS,
        ARG_PAGED_SSM_CACHE_INTERVAL,
    };
    return all_arguments_have_precision(config, index_arguments, index_precision);
}

bool PagedSelectiveSSMJitExecutor::accepts_shape(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getShape();
    const auto& B_shape = memory.at(ARG_PAGED_SSM_B)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_PAGED_SSM_STATE)->getDescPtr()->getShape();
    const auto& subsequence_shape = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || B_shape.isDynamic() || state_shape.isDynamic() || subsequence_shape.isDynamic()) {
        return false;
    }
    const auto& x_dims = x_shape.getStaticDims();
    const auto& B_dims = B_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    const auto& subsequence_dims = subsequence_shape.getStaticDims();
    return x_dims.size() == 3 && B_dims.size() == 3 && state_dims.size() == 4 && subsequence_dims.size() == 1 &&
           subsequence_dims[0] >= 1 && is_supported_state_size(state_dims.back());
}

PagedSelectiveSSMJitExecutor::PagedSelectiveSSMJitExecutor([[maybe_unused]] const PagedSelectiveSSMAttrs& attrs,
                                                           const MemoryArgs& memory,
                                                           ExecutorContext::CPtr context)
    : SelectiveSSMJitExecutorBase(std::move(context)) {
    OPENVINO_ASSERT(update(memory), "Failed to initialize PagedSelectiveSSM JIT executor");
}

bool PagedSelectiveSSMJitExecutor::update(const MemoryArgs& memory) {
    if (!accepts_shape(memory)) {
        return false;
    }
    const auto& x_dims = memory.at(ARG_PAGED_SSM_X)->getStaticDims();
    const auto& B_dims = memory.at(ARG_PAGED_SSM_B)->getStaticDims();
    const auto& state_dims = memory.at(ARG_PAGED_SSM_STATE)->getStaticDims();
    const auto& subsequence_dims = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getStaticDims();
    const auto precision = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getPrecision();
    const auto sequence_count = subsequence_dims[0] - 1;
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto outer_work =
        node::kernel::checked_size_product({sequence_count, x_dims[1]}, "JIT paged outer work items");
    const auto head_dim_tile = node::kernel::get_scratch_head_dim(x_dims[2], state_dims[3], outer_work, thread_count);
    const auto state_scratch_elements =
        node::kernel::checked_size_product({thread_count, head_dim_tile, state_dims[3]}, "JIT state scratch");
    const auto projection_elements =
        node::kernel::checked_size_product({B_dims[0], B_dims[1], B_dims[2]}, "JIT B/C projection");
    return configure_resources(
        {precision, state_dims[3], head_dim_tile, state_scratch_elements, projection_elements, state_dims[0], true});
}

void PagedSelectiveSSMJitExecutor::execute(const MemoryArgs& memory) {
    OPENVINO_ASSERT(m_kernels.ready(true) && m_scratch != nullptr,
                    "PagedSelectiveSSM JIT executor resources are not initialized");
    const auto& x_dims = memory.at(ARG_PAGED_SSM_X)->getStaticDims();
    const auto& B_dims = memory.at(ARG_PAGED_SSM_B)->getStaticDims();
    const auto& state_dims = memory.at(ARG_PAGED_SSM_STATE)->getStaticDims();
    const auto& subsequence_dims = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getStaticDims();
    const auto& block_indices_dims = memory.at(ARG_PAGED_SSM_BLOCK_INDICES)->getStaticDims();
    const auto& block_begins_dims = memory.at(ARG_PAGED_SSM_BLOCK_INDICES_BEGINS)->getStaticDims();
    const auto& processed_dims = memory.at(ARG_PAGED_SSM_NUM_PROCESSED_TOKENS)->getStaticDims();
    const auto& interval_dims = memory.at(ARG_PAGED_SSM_CACHE_INTERVAL)->getStaticDims();
    const auto sequence_count = subsequence_dims[0] - 1;
    OPENVINO_ASSERT(block_indices_dims.size() == 1 && block_begins_dims.size() == 1 && processed_dims.size() == 1 &&
                        interval_dims.size() == 1 && block_begins_dims[0] == sequence_count + 1 &&
                        processed_dims[0] == sequence_count && interval_dims[0] == sequence_count,
                    "PagedSelectiveSSM JIT metadata tensor lengths are inconsistent.");
    const node::kernel::PagedSelectiveSSMShape shape{x_dims[0],
                                                     x_dims[1],
                                                     x_dims[2],
                                                     B_dims[1],
                                                     state_dims[3],
                                                     state_dims[0],
                                                     block_indices_dims[0],
                                                     sequence_count};
    const auto precision = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getPrecision();
    const auto [input_projections, output_projections] =
        prepare_projections(memory.at(ARG_PAGED_SSM_B)->getData(), memory.at(ARG_PAGED_SSM_C)->getData());
    kernel::PagedSelectiveSSMJitRuntimeArgs args;
    args.state_decay_rates = memory.at(ARG_PAGED_SSM_A)->getData();
    args.time_steps = memory.at(ARG_PAGED_SSM_DT)->getData();
    args.input_projections = input_projections;
    args.input = memory.at(ARG_PAGED_SSM_X)->getData();
    args.output_projections = output_projections;
    args.state_cache = memory.at(ARG_PAGED_SSM_STATE)->getData();
    args.subsequence_begins = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getData();
    args.block_indices = memory.at(ARG_PAGED_SSM_BLOCK_INDICES)->getData();
    args.block_indices_begins = memory.at(ARG_PAGED_SSM_BLOCK_INDICES_BEGINS)->getData();
    args.num_processed_tokens = memory.at(ARG_PAGED_SSM_NUM_PROCESSED_TOKENS)->getData();
    args.cache_intervals = memory.at(ARG_PAGED_SSM_CACHE_INTERVAL)->getData();
    args.output = memory.at(ARG_PAGED_SSM_OUT)->getData();
    args.shape = shape;
    args.data_precision = precision;
    args.index_precision = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getDescPtr()->getPrecision();
    args.state_scratch = m_scratch->getDataAs<float>();
    const auto projection_scratch_elements =
        precision == ov::element::f32
            ? size_t{0}
            : node::kernel::checked_size_product({size_t{2}, m_requirements.projection_elements},
                                                 "JIT B/C projection scratch");
    args.metadata_validation_scratch = reinterpret_cast<int32_t*>(
        args.state_scratch + m_requirements.state_scratch_elements + projection_scratch_elements);
    args.head_dim_tile = m_requirements.head_dim_tile;
    args.cpu_parallel = m_context->getCpuParallel();
    args.fp32_state_kernel = m_kernels.fp32_state.get();
    args.direct_state_kernel = m_kernels.direct_state.get();
    args.no_state_store_kernel = m_kernels.no_state_store.get();
    kernel::paged_selective_ssm_jit(args);
}

}  // namespace ov::intel_cpu
