// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/selective_ssm.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {
namespace {

constexpr size_t target_scratch_elements = 8192;

}  // namespace

PagedSelectiveSSM::PagedSelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void PagedSelectiveSSM::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    const auto data_precision = getOriginalInputPrecisionAtPort(0);
    OPENVINO_ASSERT(any_of(data_precision, ov::element::f32, ov::element::f16, ov::element::bf16),
                    "PagedSelectiveSSM supports only f32/f16/bf16 data, got ",
                    data_precision,
                    ".");
    for (size_t port = 1; port <= 5; ++port) {
        OPENVINO_ASSERT(getOriginalInputPrecisionAtPort(port) == data_precision,
                        "PagedSelectiveSSM requires one data precision on ports 0..5.");
    }
    const auto index_precision = getOriginalInputPrecisionAtPort(6);
    OPENVINO_ASSERT(any_of(index_precision, ov::element::i32, ov::element::i64),
                    "PagedSelectiveSSM supports only i32/i64 metadata, got ",
                    index_precision,
                    ".");
    for (size_t port = 7; port <= 10; ++port) {
        OPENVINO_ASSERT(getOriginalInputPrecisionAtPort(port) == index_precision,
                        "PagedSelectiveSSM requires one metadata precision on ports 6..10.");
    }

    std::vector<PortConfigurator> input_configs;
    input_configs.reserve(11);
    for (size_t port = 0; port <= 5; ++port) {
        input_configs.emplace_back(LayoutType::ncsp, data_precision, getInputShapeAtPort(port), false, -1);
    }
    for (size_t port = 6; port <= 10; ++port) {
        input_configs.emplace_back(LayoutType::ncsp, index_precision, getInputShapeAtPort(port), false, -1);
    }
    std::vector<PortConfigurator> output_configs = {
        PortConfigurator{LayoutType::ncsp, data_precision, getOutputShapeAtPort(0), false, -1}};
    addSupportedPrimDesc(input_configs, output_configs, impl_desc_type::ref_any);
}

void PagedSelectiveSSM::update_scratchpad() {
    const auto& x_shape = getSrcMemoryAtPort(3)->getDescPtr()->getShape();
    const auto& state_shape = getSrcMemoryAtPort(5)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || state_shape.isDynamic()) {
        return;
    }
    const auto& x_dims = x_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 3 && state_dims.size() == 4);
    const auto head_dim = x_dims[2];
    const auto state_size = state_dims[3];
    const auto physical_blocks = state_dims[0];
    OPENVINO_ASSERT(state_size > 0, "PagedSelectiveSSM state_size must be greater than zero.");
    const auto scratch_head_dim = std::max(size_t{1}, std::min(head_dim, target_scratch_elements / state_size));
    if (m_state_scratch && m_block_owners && m_scratch_head_dim == scratch_head_dim &&
        m_scratch_state_size == state_size && m_cached_physical_blocks == physical_blocks) {
        return;
    }

    const auto thread_count = static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads());
    const auto state_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{thread_count, scratch_head_dim * state_size});
    const auto owner_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32,
                                               ov::intel_cpu::Shape{std::max(size_t{1}, physical_blocks)});
    m_state_scratch = context->getScratchPad()->createScratchPadMem(state_desc);
    m_block_owners = context->getScratchPad()->createScratchPadMem(owner_desc);
    m_scratch_head_dim = scratch_head_dim;
    m_scratch_state_size = state_size;
    m_cached_physical_blocks = physical_blocks;
}

void PagedSelectiveSSM::createPrimitive() {
    update_scratchpad();
}

void PagedSelectiveSSM::prepareParams() {
    update_scratchpad();
}

void PagedSelectiveSSM::execute([[maybe_unused]] const dnnl::stream& strm) {
    update_scratchpad();
    OPENVINO_ASSERT(m_state_scratch && m_block_owners);

    const auto& dt_dims = getSrcMemoryAtPort(1)->getStaticDims();
    const auto& B_dims = getSrcMemoryAtPort(2)->getStaticDims();
    const auto& x_dims = getSrcMemoryAtPort(3)->getStaticDims();
    const auto& state_dims = getSrcMemoryAtPort(5)->getStaticDims();
    const auto& subsequence_dims = getSrcMemoryAtPort(6)->getStaticDims();
    const auto& block_indices_dims = getSrcMemoryAtPort(7)->getStaticDims();
    const auto& block_begins_dims = getSrcMemoryAtPort(8)->getStaticDims();
    const auto& processed_dims = getSrcMemoryAtPort(9)->getStaticDims();
    const auto& interval_dims = getSrcMemoryAtPort(10)->getStaticDims();
    OPENVINO_ASSERT(dt_dims.size() == 2 && B_dims.size() == 3 && x_dims.size() == 3 && state_dims.size() == 4);
    OPENVINO_ASSERT(subsequence_dims.size() == 1 && block_indices_dims.size() == 1 && block_begins_dims.size() == 1 &&
                    processed_dims.size() == 1 && interval_dims.size() == 1);
    OPENVINO_ASSERT(subsequence_dims[0] >= 1);
    const auto sequence_count = subsequence_dims[0] - 1;
    OPENVINO_ASSERT(block_begins_dims[0] == sequence_count + 1 && processed_dims[0] == sequence_count &&
                        interval_dims[0] == sequence_count,
                    "PagedSelectiveSSM metadata tensor lengths are inconsistent.");

    kernel::PagedSelectiveSSMShape shape{x_dims[0],
                                         x_dims[1],
                                         x_dims[2],
                                         B_dims[1],
                                         B_dims[2],
                                         state_dims[0],
                                         block_indices_dims[0],
                                         sequence_count};
    const auto precision = getSrcMemoryAtPort(0)->getDescPtr()->getPrecision();
    const auto index_precision = getSrcMemoryAtPort(6)->getDescPtr()->getPrecision();
    kernel::paged_selective_ssm(getSrcMemoryAtPort(0)->getData(),
                                getSrcMemoryAtPort(1)->getData(),
                                getSrcMemoryAtPort(2)->getData(),
                                getSrcMemoryAtPort(3)->getData(),
                                getSrcMemoryAtPort(4)->getData(),
                                getSrcMemoryAtPort(5)->getData(),
                                getSrcMemoryAtPort(6)->getData(),
                                getSrcMemoryAtPort(7)->getData(),
                                getSrcMemoryAtPort(8)->getData(),
                                getSrcMemoryAtPort(9)->getData(),
                                getSrcMemoryAtPort(10)->getData(),
                                getDstMemoryAtPort(0)->getData(),
                                shape,
                                precision,
                                index_precision,
                                m_state_scratch->getDataAs<float>(),
                                m_scratch_head_dim,
                                m_block_owners->getDataAs<int32_t>(),
                                context->getCpuParallel());
}

bool PagedSelectiveSSM::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                             std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::PagedSelectiveSSM>(op)) {
        errorMessage = "Node is not an instance of ov::op::internal::PagedSelectiveSSM.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
